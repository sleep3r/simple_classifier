import albumentations as albu
import numpy as np


def to_tensor(x: np.ndarray, **kwargs) -> np.ndarray:
    """
    Transposes an image array to the required shape for pytorch.

    Args:
        x (np.ndarray): image array

    Returns:
        np.ndarray: transposed image array
    """
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(image_size: int) -> albu.Compose:
    """
    Constructs preprocessing transform.

    Args:
        image_size: size of the image to transform;

    Returns:
        albu.Compose: preprocessing transform
    """
    return albu.Compose(
        [
            albu.Resize(image_size, image_size),
            albu.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            albu.Lambda(image=to_tensor),
        ],
    )
