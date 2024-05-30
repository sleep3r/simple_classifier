# Standard Library
import logging
from pathlib import Path
from typing import Callable, Literal, Union

import pandas as pd
from sklearn.metrics import classification_report
import timm
import torch
from torch.utils.data import DataLoader

# simple_classifier
from checkpoint import convert_model_onnx, load_checkpoint, save_checkpoint
from config import DLConfig, config_entrypoint, object_from_dict
from dataset import ImageDataset
from transform import get_preprocessing, get_training_augmentation, get_validation_augmentation
from utils.experiment import dump_artifacts, prepare_exp
from utils.progress import inference_progress


def load_model(cfg: DLConfig) -> torch.nn.Module:
    model = timm.create_model(cfg.model.name, pretrained=True)
    model.reset_classifier(num_classes=len(cfg.classes))
    model.cuda() if torch.cuda.is_available() else model.cpu()
    return model


def prepare_dataset_and_loader(
    cfg: DLConfig,
    dataset_path: Union[Path, str],
    split: Literal["train", "validation", "test", "quality_test"],
):
    is_train = split == "train"
    image_size = cfg.image_params.size
    level = cfg.training.augs.level
    preprocessing_transform = get_preprocessing()

    if is_train:
        augmentation = get_training_augmentation(image_size, level=level)
        imgs_per_epoch = cfg.dataset.imgs_per_epoch
    else:
        augmentation = get_validation_augmentation(image_size)
        imgs_per_epoch = None

    dataset = ImageDataset(
        dataset_path=dataset_path,
        classes=cfg.classes,
        split=split,
        augmentation=augmentation,
        preprocessing=preprocessing_transform,
        imgs_per_epoch=imgs_per_epoch,
    )
    loader_cfg = cfg.loader.train if is_train else cfg.loader.validation
    loader = DataLoader(dataset, **loader_cfg)
    return dataset, loader


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: Callable,
    train_loader: DataLoader,
    logger: logging.Logger,
):
    model.train()
    running_loss = 0.0
    for batch in inference_progress(train_loader, logger=logger, desc="Train:"):
        inputs, targets = batch
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss_value = loss(outputs, targets)
        loss_value.backward()
        optimizer.step()

        running_loss += loss_value.item()
    return running_loss


def valid_epoch(
    model: torch.nn.Module,
    loss: Callable,
    valid_loader: DataLoader,
    classes: list,
    logger: logging.Logger,
) -> dict:
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        predictions = []
        references = []
        for batch in inference_progress(valid_loader, logger=logger, desc="Valid:"):
            inputs, targets = batch
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss_value = loss(outputs, targets)
            running_loss += loss_value.item()

            predictions += preds.tolist()
            references += targets.tolist()
    # Calculate average validation loss and metrics
    report = classification_report(
        references, predictions, target_names=classes, zero_division=0, output_dict=True
    )
    valid_logs = {"loss": running_loss / len(valid_loader), **report["weighted avg"]}
    return valid_logs


def eval_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    classes: list,
    logger: logging.Logger,
) -> dict:
    model.eval()
    with torch.no_grad():
        predictions = []
        references = []
        for batch in inference_progress(loader, logger=logger, desc="Eval:"):
            inputs, targets = batch
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            predictions += preds.tolist()
            references += targets.tolist()
    report = classification_report(
        references, predictions, target_names=classes, zero_division=0, output_dict=True
    )
    return report


def train(cfg: DLConfig, meta: dict, logger: logging.Logger) -> torch.nn.Module:
    logger.info("Model initialization...")
    model = load_model(cfg)

    logger.info("Dataset initialization...")
    train_dataset, train_loader = prepare_dataset_and_loader(
        cfg, cfg.training.dataset_dir, "train"
    )
    valid_dataset, valid_loader = prepare_dataset_and_loader(
        cfg, cfg.training.dataset_dir, "validation"
    )

    logger.info(f"Loaded - train: {len(train_dataset.samples)} files")
    logger.info(f"Loaded - validation: {len(valid_dataset.samples)} files")

    loss = object_from_dict(cfg.loss)
    loss.cuda()
    optimizer = object_from_dict(cfg.optimizer, params=model.parameters())
    scheduler = object_from_dict(cfg.lr_scheduler, optimizer=optimizer)

    try:
        logger.info(f'Start running, host: {meta["host_name"]}, exp_dir: {meta["exp_dir"]}\n')
        max_score = 0.0
        for epoch_num in range(0, cfg.training.total_epochs):
            logger.info("\nEpoch: {}".format(epoch_num))
            running_loss = train_epoch(model, optimizer, loss, train_loader, logger)
            logger.info(f"Training Loss: {running_loss / len(train_loader)}")
            logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']}")

            if (epoch_num + 1) % cfg.training.eval_freq == 0:
                valid_logs = valid_epoch(model, loss, valid_loader, cfg.classes, logger)

                # Logging validation metrics
                pretty_logs = "\n".join([f"{k}: {v}" for k, v in valid_logs.items()])
                logger.info(f"Valid Logs ------------------------------\n{pretty_logs}")

                # Updating learning rate
                f1_score = valid_logs["f1-score"]
                scheduler.step(f1_score)

                if max_score < f1_score:
                    max_score = f1_score
                    meta["best_model_score"] = f1_score
                    meta["best_model_epoch"] = epoch_num

                    save_checkpoint(
                        model,
                        meta["exp_dir"] / "best_model.pth",
                        meta=meta,
                        optimizer=optimizer,
                    )
                    logger.info("Model saved!")

            if cfg.dataset.imgs_per_epoch:
                # Update the epoch_offset after each epoch
                train_dataset.update_epoch_offset()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    return model


def evaluate(model: torch.nn.Module, cfg: DLConfig, meta: dict, logger: logging.Logger):
    logger.info("Start running evaluation..")

    logger.info("Model initialization...")
    load_checkpoint(
        model,
        str(meta["exp_dir"] / "best_model.pth"),
        strict=True,
    )

    if torch.cuda.is_available():
        model.cuda()

    logger.info("Dataset initialization...")

    test_dataset, test_loader = prepare_dataset_and_loader(
        cfg, cfg.training.dataset_dir, "validation"
    )

    logger.info(f"Loaded - test: {len(test_dataset.samples)} files")

    report = eval_epoch(model, test_loader, cfg.classes, logger)
    pd.DataFrame(report).transpose().to_csv(meta["exp_dir"] / "test_metrics.csv")


def main(cfg: DLConfig) -> None:
    meta, logger = prepare_exp(cfg)

    logger.info(f"Config:\n{cfg.pretty_text}\n")

    model = train(cfg, meta, logger)

    evaluate(model, cfg, meta, logger)

    convert_model_onnx(model, model_path=meta["exp_dir"], image_size=cfg.image_params.size)

    dump_artifacts(cfg, meta)


if __name__ == "__main__":
    config: DLConfig = config_entrypoint()
    main(config)
