# Standard Library
from typing import Literal

import albumentations as albu
import cv2
import numpy as np


def get_training_augmentation(
    image_size: int,
    level: Literal["high", "low"],
) -> albu.Compose:
    """
    Constructs augmentation transform for training images.

    Returns:
        albu.Compose: augmentation transform
    """
    assert level in ("high", "low"), "Supported augmentation levels: high, low"

    if level == "low":
        return get_validation_augmentation(image_size)

    train_transform = [
        albu.Resize(image_size, image_size),
        albu.OneOf(
            [
                albu.CoarseDropout(
                    min_holes=4,
                    max_holes=6,
                    max_height=image_size // 10,
                    max_width=image_size // 10,
                ),
                albu.RandomSunFlare(
                    flare_roi=(0, 0, 1, 1),
                    num_flare_circles_lower=1,
                    num_flare_circles_upper=4,
                    src_radius=image_size // 4,
                    p=0.05,
                ),
            ],
            p=0.3,
        ),
        albu.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=5,
            border_mode=cv2.BORDER_CONSTANT,
            interpolation=cv2.INTER_LANCZOS4,
            p=0.3,
        ),
        albu.OneOf([albu.HueSaturationValue(p=0.5), albu.ToGray(p=0.2)]),
        albu.OneOf(
            [
                albu.Perspective(scale=(0.01, 0.1), p=0.5),
                albu.Affine(scale=(0.8, 1.2), p=0.5),
                albu.ElasticTransform(
                    p=0.5,
                    alpha=image_size * 1.2,
                    sigma=image_size * 1.2 * 0.05,
                    alpha_affine=image_size * 1.2 * 0.03,
                    border_mode=cv2.BORDER_REFLECT,
                ),
                albu.GridDistortion(p=0.5, distort_limit=0.5, border_mode=cv2.BORDER_REFLECT),
                albu.OpticalDistortion(
                    p=0.5, distort_limit=0.5, shift_limit=0.5, border_mode=cv2.BORDER_REFLECT
                ),
            ],
            p=0.3,
        ),
        albu.CLAHE(p=0.4),
        albu.RandomBrightnessContrast(
            brightness_limit=0.5,
            contrast_limit=0.5,
            p=0.5,
        ),
        albu.GaussNoise(p=0.2),
        albu.GaussianBlur(p=0.1, blur_limit=(3, 5)),
        albu.RandomGamma(gamma_limit=(85, 115), p=0.3),
        albu.RandomShadow(p=0.05),
        albu.OneOf(
            [
                albu.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.2), p=0.5),
                albu.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.5),
            ],
            p=0.3,
        ),
        albu.ImageCompression(quality_lower=50, p=0.3),
        albu.Downscale(scale_min=0.75, scale_max=0.99, p=0.1),
        albu.RandomRotate90(p=0.5),
        albu.HorizontalFlip(p=0.5),
        albu.Rotate(limit=20, p=0.2),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation(image_size: int) -> albu.Compose:
    """
    Constructs augmentation transform for validation images.

    Returns:
        albu.Compose: augmentation transform
    """
    validation_transform = [
        albu.Resize(image_size, image_size),
    ]
    return albu.Compose(validation_transform)


def to_tensor(x: np.ndarray, **kwargs) -> np.ndarray:
    """
    Transposes an image array to the required shape for pytorch.

    Args:
        x (np.ndarray): image array

    Returns:
        np.ndarray: transposed image array
    """
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing() -> albu.Compose:
    """
    Constructs preprocessing transform.

    Returns:
        albu.Compose: preprocessing transform
    """
    preprocessing_transform = [
        albu.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(preprocessing_transform)
