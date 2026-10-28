"""Inference backends (torch / onnx / tensorrt) behind the InferenceBackend interface."""

from perception.backends.torch_backend import TorchBackend

__all__ = ["TorchBackend"]
