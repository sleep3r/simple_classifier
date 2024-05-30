# Standard Library
import json
import os
from typing import Any, Dict


def read_json(file_path: os.PathLike) -> Dict[str, Any]:
    """
    Reads a JSON file and return its contents as a dictionary.

    Args:
        file_path (os.PathLike): path to the JSON file to read.

    Returns:
        Dict[str, Any]: dictionary containing the contents of the JSON file.
    """
    with open(file_path) as f:
        return json.load(f)
