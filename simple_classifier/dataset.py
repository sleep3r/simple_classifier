# Standard Library
import math
import os
from pathlib import Path
from typing import Callable, Literal, Optional, Union

# ML
from PIL import Image, ImageFile
import numpy as np
from torch.utils.data import Dataset

Image.MAX_IMAGE_PIXELS = None
MAX_PIXEL_SIZE = 2480 * 3508
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load image in a production manner.

    Args:
        image_path (Union[str, Path]): absolute path to the image file.

    Returns:
        np.ndarray: loaded image tensor.
    """
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    width, height = image.size
    if width * height > MAX_PIXEL_SIZE:
        factor = math.sqrt(width * height / MAX_PIXEL_SIZE)
        image = image.resize(
            size=(
                math.floor(width / factor),
                math.floor(height / factor),
            ),
            resample=Image.LANCZOS,
        )
    return np.asarray(image, np.uint8)


class ImageDataset(Dataset):
    def __init__(
        self,
        dataset_path: Union[Path, str],
        classes: list[str],
        split: Literal["train", "validation", "test", "quality_test"],
        augmentation: Optional[Callable] = None,
        preprocessing: Optional[Callable] = None,
        imgs_per_epoch: Optional[int] = None,
    ):
        """
        Dataset for loading image for a document classification task.

        Args:
            dataset_path (Path): path to directory containing the input images;
            classes (List[str]): list of class names to include in the masks;
            split (Literal["train", "validation", "test", "quality_test"]): dataset split to use;
            augmentation (Optional[Callable]): optional callable for image augmentation;
            preprocessing (Optional[Callable]): optional callable for preprocessing the images;
            imgs_per_epoch (Optional[int]): number of images to use per epoch during training.
        """
        self.images_path = os.path.join(dataset_path, split)
        self.classes = classes
        self.class_indexes = [self.classes.index(cls) for cls in classes]
        self.split = split
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.imgs_per_epoch = imgs_per_epoch
        self.epoch_offset = 0

        self.samples = self._make_dataset()

    def _make_dataset(self):
        """
        Generates a list of samples of a form (path_to_sample, class).

        Raises:
            FileNotFoundError: If ``self.images_path`` has no class folders.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        instances = []
        available_classes = set()
        for target_class, class_index in zip(self.classes, self.class_indexes):
            target_dir = os.path.join(self.images_path, target_class)
            if not os.path.isdir(target_dir):
                continue

            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

        empty_classes = set(self.classes) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}."
            raise FileNotFoundError(msg)

        return instances

    def __len__(self):
        if self.split == "train" and self.imgs_per_epoch is not None:
            return self.imgs_per_epoch
        return len(self.samples)

    def update_epoch_offset(self) -> None:
        """
        Updates the epoch offset for the dataset.
        We use fixed size epochs with continuous training through all images.
        """
        if self.imgs_per_epoch is not None:
            self.epoch_offset += self.imgs_per_epoch
            if self.epoch_offset >= len(self.samples):
                self.epoch_offset %= len(self.samples)

    def __getitem__(self, i):
        # Recalculate index
        i = (i + self.epoch_offset) % len(self.samples)

        # Load image
        path, class_index = self.samples[i]
        image = load_image(path)

        # Apply augmentation if provided
        if self.augmentation:
            image = self.augmentation(image=image)["image"]

        # Apply preprocessing if provided
        if self.preprocessing:
            image = self.preprocessing(image=image)["image"]
        return image, class_index
