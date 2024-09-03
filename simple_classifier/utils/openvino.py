# Standard Library
from pathlib import Path

from openvino.runtime import Core
from openvino.runtime.ie_api import CompiledModel


def load_openvino_model(model_path: Path) -> tuple[CompiledModel, str, str]:
    """
    Loads an OpenVINO model from the given path.

    Args:
        model_path (Path): path to the model file. It can be either
        a directory containing .xml and .bin files, or a .onnx file.

    Returns:
        tuple[CompiledModel, str, str]: loaded OpenVINO model, the input node name
        and the output node name.

    Raises:
        FileNotFoundError: raised when the model format is not supported
        or the files don't exist.
    """
    core = Core()

    is_directory = model_path.is_dir()
    xml_exists = (model_path / "model.xml").exists()
    bin_exists = (model_path / "model.bin").exists()

    if is_directory and xml_exists and bin_exists:
        net = core.read_model(model_path / "model.xml", model_path / "model.bin")
    elif model_path.is_file() and model_path.name.endswith(".onnx"):
        net = core.read_model(model_path)
    else:
        raise FileNotFoundError("Model's format is not supported!")

    model = core.compile_model(model=net, device_name="CPU")
    input_node_name = next(iter(model.inputs))
    out_node_name = next(iter(model.outputs))
    return model, input_node_name, out_node_name
