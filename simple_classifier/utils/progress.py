# Standard Library
import logging
from typing import Iterable

from tqdm import tqdm

# simple_classifier
from utils.log import TqdmToLogger


def inference_progress(iterable: Iterable, logger: logging.Logger, desc: str) -> tqdm:
    return tqdm(
        iterable,
        desc=desc,
        unit="batch",
        file=TqdmToLogger(logger, logging.WARNING),
        mininterval=1,
    )
