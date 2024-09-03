# Standard Library
from os import PathLike
from pathlib import Path
from typing import Callable, Literal

import numpy as np

# simple_classifier
from simple_classifier.config import DLConfig
from simple_classifier.utils.postprocessing import softmax
from simple_classifier.utils.preprocessing import get_preprocessing

try:
    # simple_classifier
    from simple_classifier.utils.openvino import load_openvino_model  # noqa: WPS433
except ImportError:
    load_openvino_model = None  # type: ignore
try:
    # simple_classifier
    from simple_classifier.utils.onnxruntime import load_onnxruntime_model  # noqa: WPS433
except ImportError:
    load_onnxruntime_model = None  # type: ignore


class Model:
    def __init__(
        self,
        model_dir: PathLike | str,
        device: Literal["cpu", "gpu"] = "cpu",
        temperature: float = 1.0,
    ):
        """
        Inference model for image classification task.

        Args:
            model_dir (os.PathLike | str): path to the directory containing
            the model and configuration files;
            device (Literal["cpu", "gpu"]): processor type for inference.
            In the case of cpu, OpenVINO will be used for inference. In case of gpu -
            OnnxRuntime.
            temperature (float): temperature value for softmax function.
        """
        self.model_path = Path(model_dir)
        self.cfg = DLConfig.load(self.model_path / "config.yml")
        self.device = device
        (  # noqa: WPS414
            self.classifier,
            self._input_blob,
            self._out_blob,
        ) = self._load_model()
        self.temperature = temperature

        self.preprocessing = get_preprocessing(image_size=self.cfg.image_params.size)

    def predict(self, image: np.ndarray) -> tuple[str, float, dict[str, float]]:
        """
        Predicts class for a given image.

        Args:
            image (np.ndarray): input image as a numpy array.

        Returns:
            tuple: tuple of predicted label, max prob and all class probs as a dict
                for given image.

        """
        labels, probs, classes_probs = self.predict_batch([image])
        return labels[0], probs[0], classes_probs[0]

    def predict_batch(  # noqa: WPS234
        self,
        images: list[np.ndarray],
    ) -> tuple[list[str], list[float], list[dict[str, float]]]:
        """
        Predicts classes for a given image list.

        Args
            images (list[np.ndarray]): batch images as a list of numpy arrays
        Returns:
            tuple: tuple of predicted labels, max probs and all class probs as a dict
                for each image in batch
        """
        image_tensors = [self.preprocessing(image=image)["image"] for image in images]
        x_tensor = np.stack(image_tensors, axis=0)
        outputs = self._infer_model(x_tensor)
        full_probs = softmax(outputs, self.temperature)

        indices = np.argmax(full_probs, axis=1).astype(np.uint8)
        probs = full_probs[np.arange(len(full_probs)), indices]
        labels = [self.cfg.classes[i] for i in indices]
        classes_probs = [dict(zip(self.cfg.classes, p)) for p in full_probs]
        return labels, probs, classes_probs

    def _load_model(self) -> tuple[Callable, str, str]:
        """
        Loads a model with OpenVINO or ONNXRuntime inference framework.
        Uses OpenVINO with cpu device and ONNXRuntime with gpu.

        Returns:
            tuple[CompiledModel, str, str]: loaded model, the input node name
            and the output node name.

        Raises:
            ValueError: raised when importing specific inference framework library for
            selected device failed.
        """
        if self.device == "gpu" and load_onnxruntime_model is not None:
            classifier, input_blob, out_blob = load_onnxruntime_model(
                self.model_path / "model.onnx",
            )
        elif self.device == "cpu" and load_openvino_model is not None:
            classifier, input_blob, out_blob = load_openvino_model(
                self.model_path / "model.onnx"
            )
        else:
            raise ValueError("Model cannot be loaded! Check the device and requirements")
        return classifier, input_blob, out_blob

    def _infer_model(self, x: np.ndarray) -> np.ndarray:
        """
        Infer model.

        Args
            x (np.ndarray): prepared batch tensor as numpy array.
        Returns:
            np.ndarray: model output as numpy array.
        """
        inputs = {self._input_blob: x}
        if self.device == "cpu":
            outputs = self.classifier.infer_new_request(inputs=inputs)  # type: ignore
            outputs = outputs[self._out_blob]
        else:
            outputs = self.classifier.run([self._out_blob], inputs)[0]  # type: ignore
        return outputs
