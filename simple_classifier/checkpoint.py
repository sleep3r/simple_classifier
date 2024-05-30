# Standard Library
from collections import OrderedDict
import logging
from os import PathLike
from pathlib import Path
import re
from typing import Any, Optional, Union

import torch
from torch.optim import Optimizer


def load_state_dict(
    module: torch.nn.Module,
    state_dict: OrderedDict,
    strict: bool = True,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Loads state_dict to a module.
    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): module that receives the state_dict;
        state_dict (OrderedDict): weights;
        strict (bool): whether to strictly enforce that the keys in :attr:`state_dict`
                       match the keys returned by this module's :meth:`~torch.nn.Module.state_dict`
                       function. Default: ``False``;
        logger (:obj:`logging.Logger`, optional): logger to log the error message.
                                                  If not specified, print function will be used.
    """
    unexpected_keys: list = []
    all_missing_keys: list = []
    err_msg: list = []

    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata  # type: ignore

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=""):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if hasattr(module, "module"):
            module = module.module

        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            all_missing_keys,
            unexpected_keys,
            err_msg,
        )

        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(module)
    load = None  # type: ignore # break load -> load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [key for key in all_missing_keys if "num_batches_tracked" not in key]

    if unexpected_keys:
        err_msg.append(
            'unexpected key in source state_dict: {", ".join(unexpected_keys)}\n',
        )
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n',
        )

    if len(err_msg) > 0:
        err_msg.insert(0, "The model and loaded state dict do not match exactly\n")
        err_msg = "\n".join(err_msg)  # type: ignore
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)


def load_checkpoint(
    model: torch.nn.Module,
    filename: str,
    map_location: Optional[str] = None,
    strict: bool = True,
    logger: Optional[logging.Logger] = None,
    revise_keys: list = [(r"^module\.", "")],
) -> dict:
    """
    Loads checkpoint from file.

    Args:
        model (Module): module to load checkpoint;
        filename (str): accept local filepath;
        map_location (str): same as :func:`torch.load`;
        strict (bool): whether to allow different params for the model and checkpoint;
        logger (:mod:`logging.Logger` or None): the logger for error message;
        revise_keys (list): a list of customized keywords to modify the
                            state_dict in checkpoint. Each item is a (pattern, replacement)
                            pair of the regular expression operations. Default: strip
                            the prefix 'module.' by [(r'^module\\.', '')].

    Returns:
        dict or OrderedDict: the loaded checkpoint.
    """
    checkpoint = torch.load(filename, map_location=map_location)
    # OrderedDict is a subclass of a dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")

    # get state_dict from checkpoint
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    # strip prefix of state_dict
    for p, r in revise_keys:
        state_dict = {re.sub(p, r, k): v for k, v in state_dict.items()}

    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def weights_to_cpu(state_dict: dict) -> dict:
    """
    Copies a model state_dict to cpu.

    Args:
        state_dict (dict): model weights on GPU.

    Returns:
        dict: model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    # Keep metadata in state_dict
    state_dict_cpu._metadata = getattr(state_dict, "_metadata", OrderedDict())  # type: ignore
    return state_dict_cpu


def _save_to_state_dict(
    module: torch.nn.Module,
    destination: dict,
    prefix: str,
    keep_vars: bool,
) -> None:
    """
    Saves module state to `destination` dictionary.
    This method is modified from :meth:`torch.nn.Module._save_to_state_dict`.

    Args:
        module (nn.Module): The module to generate state_dict;
        destination (dict): A dict where state will be stored;
        prefix (str): The prefix for parameters and buffers used in this
            module.
    """
    for name, param in module._parameters.items():
        if param is not None:
            destination[prefix + name] = param if keep_vars else param.detach()
    for name, buf in module._buffers.items():
        # remove check of _non_persistent_buffers_set to allow nn.BatchNorm2d
        if buf is not None:
            destination[prefix + name] = buf if keep_vars else buf.detach()


def get_state_dict(
    module: torch.nn.Module,
    destination: Optional[dict[Any, Any]] = None,
    prefix: str = "",
    keep_vars: bool = False,
) -> dict:
    """
    Returns a dictionary containing a whole state of the module.
    Both parameters and persistent buffers (e.g. running averages) are
    included. Keys are corresponding parameter and buffer names.
    This method is modified from :meth:`torch.nn.Module.state_dict` to
    recursively check parallel module in case that the model has a complicated
    structure, e.g., nn.Module(nn.Module(DDP)).

    Args:
        module (nn.Module): module to generate state_dict;
        destination (OrderedDict): returned dict for the state of the module;
        prefix (str): prefix of the key;
        keep_vars (bool): whether to keep the variable property of the parameters. Default: False.

    Returns:
        dict: dictionary containing a whole state of the module.
    """
    # recursively check parallel module in case that the model has a
    # complicated structure, e.g., nn.Module(nn.Module(DDP))
    if hasattr(module, "module"):
        module = module.module  # type: ignore

    # below is the same as torch.nn.Module.state_dict()
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()  # type: ignore

    destination._metadata[prefix[:-1]] = local_metadata = {  # type: ignore
        "version": module._version
    }
    _save_to_state_dict(module, destination, prefix, keep_vars)

    for name, child in module._modules.items():
        if child is not None:
            get_state_dict(child, destination, prefix + name + ".", keep_vars=keep_vars)

    for hook in module._state_dict_hooks.values():
        hook_result = hook(module, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination  # type: ignore


def save_checkpoint(
    model: torch.nn.Module,
    filename: Union[PathLike, str],
    meta: Optional[dict[str, Any]] = None,
    optimizer=None,
) -> None:
    """
    Saves checkpoint to file.
    The checkpoint will have 3 fields: ``state_dict`` and
    ``optimizer`` and ``meta``. By default, ``meta`` will contain version and time info.

    Args:
        model (Module): module whose params are to be saved;
        filename (str): checkpoint filename.
        optimizer (:obj:`Optimizer`, optional): optimizer to be saved.
    """
    if hasattr(model, "module"):
        model = model.module  # type: ignore

    checkpoint = {"meta": meta, "state_dict": weights_to_cpu(get_state_dict(model))}
    # save optimizer state dict in the checkpoint
    if isinstance(optimizer, Optimizer):
        checkpoint["optimizer"] = optimizer.state_dict()
    elif isinstance(optimizer, dict):
        checkpoint["optimizer"] = {}
        for name, optim in optimizer.items():
            checkpoint["optimizer"][name] = optim.state_dict()  # type: ignore

    torch.save(checkpoint, filename)


def convert_model_onnx(model: torch.nn.Module, model_path: Path, image_size: int) -> None:
    """
    Converts a PyTorch model to ONNX format.

    Args:
        model (torch.nn.Module): PyTorch model to be converted;
        model_path (Path): path where the ONNX model will be saved;
        image_size (int): size of the input images for the model.

    Returns:
        None
    """
    # load best model
    load_checkpoint(
        model,
        str(model_path / "best_model.pth"),
        strict=True,
    )

    # ensure the model is in CPU
    model.to("cpu")
    input_tensor = torch.randn(1, 3, image_size, image_size)

    # ensure the input tensor is also in CPU
    input_tensor = input_tensor.to("cpu")

    # export onnx
    torch.onnx.export(
        model,
        input_tensor,
        model_path / "model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": [0],
            "output": [0],
        },
    )
