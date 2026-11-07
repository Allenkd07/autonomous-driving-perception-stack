"""TensorRT backend: runs a serialized engine built from the ONNX export by scripts/build_engine.py.

An engine is the network compiled for one GPU model and one TensorRT version: TensorRT picks the fastest
kernel for every layer on that GPU while building, fuses layers, and with FP16 runs most layers in half
precision. So engines are built on the machine that runs them and aren't committed.

GPU buffers are PyTorch tensors, which keeps the backend free of extra CUDA bindings (pycuda / cuda-python)
and shares PyTorch's CUDA context and memory pool with the other stages.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from perception.core.registry import register


@register("backend", "tensorrt")
class TensorRTBackend:
    """TensorRT engine behind the InferenceBackend interface.

    model    .engine file (scripts/build_engine.py)
    inputs   input names, as in the ONNX graph the engine was built from
    outputs  output names, as in the ONNX graph
    device   "cuda" or "cuda:N"; None is "cuda"
    half     ignored: the precision was chosen when the engine was built
    Inputs and outputs are float32 numpy arrays (an FP16 engine keeps float32 I/O and converts inside).
    """

    def __init__(
        self,
        model: str | Path,
        inputs: Sequence[str] = ("images",),
        outputs: Sequence[str] = ("features",),
        device: str | None = None,
        half: bool = False,
    ) -> None:
        import tensorrt as trt
        import torch

        self._torch = torch
        self.device = torch.device(device or "cuda")
        if self.device.type != "cuda":
            raise ValueError(f"TensorRT runs on CUDA devices only, not {self.device}")
        logger = trt.Logger(trt.Logger.WARNING)
        with torch.cuda.device(self.device):
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(Path(model).read_bytes())
            if self.engine is None:
                raise RuntimeError(f"could not load {model}: built for another TensorRT version or GPU?")
            self.context = self.engine.create_execution_context()
            self.stream = torch.cuda.Stream(self.device)

        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        missing = (set(inputs) | set(outputs)) - set(names)
        if missing:
            raise ValueError(f"{model} has no tensor(s) {sorted(missing)}: it has {names}")
        self.inputs = tuple(inputs)
        self.outputs = tuple(outputs)
        self.dtypes = {
            name: getattr(torch, np.dtype(trt.nptype(self.engine.get_tensor_dtype(name))).name)
            for name in names
        }

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        torch = self._torch
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            # The tensors must stay alive until the stream has run: TensorRT only holds their addresses
            buffers = {}
            for name in self.inputs:
                host = torch.from_numpy(np.ascontiguousarray(inputs[name]))
                buffers[name] = host.to(self.device, self.dtypes[name])
                self.context.set_input_shape(name, tuple(buffers[name].shape))
            for name in self.outputs:
                shape = tuple(self.context.get_tensor_shape(name))  # known once the input shapes are set
                buffers[name] = torch.empty(shape, dtype=self.dtypes[name], device=self.device)
            for name, tensor in buffers.items():
                self.context.set_tensor_address(name, tensor.data_ptr())
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT inference failed")
            out = {name: buffers[name].float().cpu() for name in self.outputs}
        self.stream.synchronize()
        return {name: t.numpy() for name, t in out.items()}


def build_engine(
    onnx: str | Path,
    out: str | Path,
    precision: str = "fp16",
    batch: tuple[int, int, int] = (1, 8, 32),
    workspace_gb: float = 1.0,
    verbose: bool = True,
) -> Path:
    """Compile an ONNX model into a TensorRT engine file for this GPU.

    precision     fp16 lets TensorRT run layers in half precision where that is faster (layers without an
                  FP16 kernel stay FP32); fp32 for a reference
    batch         (min, opt, max) for an input with a dynamic batch; TensorRT tunes its kernels for opt
    workspace_gb  scratch memory the builder may give a layer's kernel
    """
    import tensorrt as trt

    if precision not in ("fp16", "fp32"):
        raise ValueError(f"precision must be fp16 or fp32, not {precision!r}")
    onnx, out = Path(onnx), Path(out)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # TensorRT 10: explicit batch is the only mode
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"could not parse {onnx}:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    dynamic = False
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        shape = tuple(tensor.shape)
        if any(d < 0 for d in shape[1:]):
            raise ValueError(f"{tensor.name}: only the batch dimension may be dynamic, got {shape}")
        if shape[0] < 0:
            dynamic = True
            profile.set_shape(tensor.name, *[(b, *shape[1:]) for b in batch])
        if verbose:
            print(f"input {tensor.name} {shape}")
    if dynamic:
        config.add_optimization_profile(profile)

    start = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("engine build failed (see the TensorRT messages above)")
    out.write_bytes(serialized)
    if verbose:
        print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.perf_counter() - start:.0f} s")
    return out
