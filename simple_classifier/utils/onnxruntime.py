from pathlib import Path

from onnxruntime import InferenceSession


def load_onnxruntime_model(
    model_path: Path,
) -> tuple[InferenceSession, str, str]:
    """
    Loads an ONNX Runtime model from the given path.

    Args:
        model_path (Path): path to the model .onnx file.

    Returns:
        tuple[InferenceSession, str, str]: created ONNXRuntime session,
        the input node name and the output node name.

    Raises:
        FileNotFoundError: raised when the model format is not supported
        or the files don't exist.
    """
    session = InferenceSession(
        path_or_bytes=model_path,
        providers=["CUDAExecutionProvider"],
    )

    if "CUDAExecutionProvider" not in session.get_providers():
        raise ValueError("Failed to load ONNX Runtime model on gpu")

    input_node_name = session.get_inputs()[0].name
    out_node_name = session.get_outputs()[0].name
    return session, input_node_name, out_node_name
