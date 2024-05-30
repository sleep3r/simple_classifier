# Standard Library
import os
from typing import Any

# simple_classifier
from config import DLConfig


def setup_training(
    cfg: DLConfig,
    meta: dict,
    model_name: str,
    dataset_name: str,
    dataset_version: int,
) -> tuple[dict, Training]:
    """
    Sets up a new training session with MLECO
    and configures it with the given dataset and parameters.

    Args:
        cfg (DLConfig): config;
        meta (dict): meta-information dictionary;
        model_name (str): name of the MLECO mode;.
        dataset_name (str): name of the dataset;
        dataset_version (int): version of the dataset.

    Returns:
        tuple[dict, Training]: updated meta dictionary and the new training session object.
    """

    def recurse_add_param(data, prefix=""):
        for key, value in data.items():
            new_key = prefix + key
            if isinstance(value, dict):
                recurse_add_param(value, new_key + ".")
            else:
                training.add_param(new_key, value)

    model = MlecoModel(model_name)

    training = model.new_training()
    training.set_dataset(dset=dataset_name, version=dataset_version)
    training.add_param("MLECO_COMMIT_HASH", os.getenv("MLECO_COMMIT_HASH"))
    training.add_param("COMMIT_SHA", meta.get("sha"))
    recurse_add_param(cfg)

    meta["model_id"] = training.model_id
    meta["training_id"] = training.training_id
    return meta, training


def log_artifacts_mleco(meta: dict[str, Any], training: Training) -> None:
    """
    Logs artifacts and metrics for a completed training run.

    Args:
        meta (Dict[str, Any]): dictionary of metadata associated with the training run;
        training (Training): Training object associated with the completed training run.

    Returns:
        None
    """
    training.set_folder(meta["exp_dir"])

    # base
    training.add_artifacts("config.yml")  # full model config
    training.add_artifacts("report.json")  # json with main training data
    training.add_artifacts("run.log")  # log

    training.add_artifacts("best_model.pth")  # best torch checkpoint
    training.add_artifacts("model.onnx")  # onnx model weights

    if "best_model_score" in meta:
        training.add_metric("best_model_score", meta["best_model_score"])

    training.add_artifacts("*.csv")  # val report

    training.save()
