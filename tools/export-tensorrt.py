#!/bin/sh
# ruff: noqa: EXE003, D300, D205, T201 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 -P "$0" "$@"
Export the NomosUni upscaler to TensorRT for FFmpeg dnn_processing.

This script converts the 2x NomosUni model to TensorRT engines (.engine files)
that can be loaded by FFmpeg's TensorRT DNN backend.

Usage:
    # Show the supported model
    python export-tensorrt.py --list

    # Export a fixed 1080p engine
    python export-tensorrt.py --min-height 1080 --opt-height 1080 \\
        --max-height 1080 -o model.engine

    # Export with custom height range
    python export-tensorrt.py --min-height 720 --max-height 1080

Requirements:
    pip install torch onnx onnxconverter-common safetensors tensorrt

Example FFmpeg usage after export:
    ffmpeg -i input.mp4 -vf "dnn_processing=dnn_backend=tensorrt:model=model.engine" output.mp4
'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast, override
from urllib import request

import argparse
import tempfile

from onnxconverter_common import float16
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

import onnx
import tensorrt as trt
import torch


def download_model(cache_dir: Path) -> Path:
    """Download the Nomos model weights.

    Args:
        cache_dir: Directory in which to reuse or save model weights.

    Returns:
        path: Local safetensors checkpoint.
    """
    filename = "2xNomosUni_compact_otf_medium.safetensors"
    model_url = (
        "https://huggingface.co/Phips/2xNomosUni_compact_otf_medium/"
        "resolve/3241c877a6e09036f9e466c840822bb066f11c44/"
        "2xNomosUni_compact_otf_medium.safetensors"
    )
    path = cache_dir / filename
    if path.exists():
        print(f"Using cached model: {path}")
        return path

    print(f"Downloading {filename}...")

    # Download to a temp file first, then rename to avoid partial downloads.
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with (
            request.urlopen(model_url, timeout=300) as response,
            open(temp_path, "wb") as f,
        ):
            f.write(response.read())
        # Verify the download succeeded and file is not empty.
        file_size = temp_path.stat().st_size
        if file_size == 0:
            raise RuntimeError(f"Downloaded file is empty: {temp_path}")
        temp_path.rename(path)
        print(f"Downloaded to {path} ({file_size / 1024 / 1024:.1f} MB)")
    except Exception as e:
        # Clean up partial download.
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to download model from {model_url}: {e}") from e

    return path


def list_models() -> None:
    """Print the supported model."""
    print(
        f"\nSupported model:\n\n  {'2x-nomosuni-compact':24s} "
        "Fast universal upscale - compression, noise, and blur handling\n",
    )


def build_nomos_model(state_dict: dict[str, torch.Tensor]) -> nn.Module:
    """Build the Nomos SRVGG network represented by the downloaded weights.

    Args:
        state_dict: Weights whose convolution count determines network depth.

    Returns:
        model: Uninitialized network matching the checkpoint architecture.
    """
    num_conv_layers = sum(
        1 for key, value in state_dict.items() if "weight" in key and len(value.shape) == 4
    )
    return _NomosSRVGG(num_conv=num_conv_layers, scale=2)


def load_model(cache_dir: Path | None = None) -> nn.Module:
    """Load the Nomos model.

    Args:
        cache_dir: Weight cache directory, or the provisioned default.

    Returns:
        model: Loaded network in evaluation mode.
    """

    if cache_dir is None:
        cache_dir = Path("/opt/scratch/caches/netv/ai_upscale")
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_path = download_model(cache_dir)
    print(f"Loading PyTorch model from {model_path}")
    state_dict = load_file(model_path, device="cpu")
    model = build_nomos_model(state_dict)

    model.load_state_dict(state_dict)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded SRVGGNetCompact ({params:.2f}M params), Scale: 2x")
    return model


def export_onnx(
    model: nn.Module,
    opt_shape: tuple[int, int],
    onnx_path: Path | str,
    *,
    fixed_shape: bool = False,
    precision: str = "fp32",
) -> None:
    """Export a model to ONNX format.

    Args:
        model: Loaded upscaler to export.
        opt_shape: Example input width and height.
        onnx_path: Destination model file.
        fixed_shape: Whether to omit dynamic spatial axes.
        precision: Requested ONNX floating-point format.
    """

    opt_w, opt_h = opt_shape
    print(f"Exporting to ONNX: {onnx_path}")
    print(f"  Optimal shape: 1x3x{opt_h}x{opt_w}")

    dummy_input = torch.randn(1, 3, opt_h, opt_w, device="cpu")

    dynamic_axes = None
    if not fixed_shape:
        dynamic_axes = {
            "input": {
                2: "height",
                3: "width",
            },
            "output": {
                2: "out_height",
                3: "out_width",
            },
        }

    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    if precision == "fp16":
        converted = float16.convert_float_to_float16(
            onnx.load(onnx_path),
            keep_io_types=False,
        )
        onnx.save(converted, onnx_path)
    elif precision == "bf16":
        print("  BF16 ONNX conversion is unavailable; TensorRT will select BF16 kernels")
    shape_mode = "fixed shape" if fixed_shape else "dynamic H/W"
    print(f"  ONNX export complete ({shape_mode}, {precision})")


def build_engine(
    onnx_path: Path | str,
    engine_path: Path | str,
    min_shape: tuple[int, int],
    opt_shape: tuple[int, int],
    max_shape: tuple[int, int],
    precision: str = "fp16",
    workspace_gb: int = 4,
    opt_level: int = 3,
) -> None:
    """Build a TensorRT engine from an ONNX model with dynamic shapes.

    Args:
        onnx_path: Input ONNX model file.
        engine_path: Destination serialized engine file.
        min_shape: Minimum input width and height.
        opt_shape: Optimal input width and height.
        max_shape: Maximum input width and height.
        precision: Requested compute and I/O precision.
        workspace_gb: Builder workspace budget in GiB.
        opt_level: TensorRT builder optimization level.
    """

    min_w, min_h = min_shape
    opt_w, opt_h = opt_shape
    max_w, max_h = max_shape

    print(f"Building TensorRT engine: {engine_path}")
    print("  Dynamic shapes:")
    print(f"    min: {min_w}x{min_h}")
    print(f"    opt: {opt_w}x{opt_h}")
    print(f"    max: {max_w}x{max_h}")
    print(f"  Precision: {precision}")
    print(f"  Workspace: {workspace_gb} GB")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    # Explicit batch is mandatory in TensorRT 10+; TensorRT 11 removed the
    # legacy flag after deprecating it in TensorRT 10.
    explicit_batch = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    network_flags = 0 if explicit_batch is None else 1 << int(explicit_batch)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    # Optimization level (0-5, default is 3)
    # Higher levels enable more aggressive kernel selection/fusion but use more memory.
    config.builder_optimization_level = opt_level
    print(f"  Optimization level: {opt_level}")

    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(
        input_name,
        min=(1, 3, min_h, min_w),
        opt=(1, 3, opt_h, opt_w),
        max=(1, 3, max_h, max_w),
    )
    config.add_optimization_profile(profile)

    # Set compute precision.
    if precision in ("fp16", "bf16"):
        fp16_flag = getattr(trt.BuilderFlag, "FP16", None)
        if fp16_flag is None:
            pass  # TensorRT 11 uses strongly typed tensor dtypes instead.
        elif getattr(builder, "platform_has_fast_fp16", True):
            config.set_flag(fp16_flag)
        else:
            print("  Warning: FP16/BF16 not supported on this platform, using FP32")
            precision = "fp32"
    if precision == "bf16":
        if hasattr(trt.BuilderFlag, "BF16"):
            config.set_flag(trt.BuilderFlag.BF16)
        else:
            print("  Warning: BF16 not supported by TensorRT, using FP16")
            precision = "fp16"

    # Set I/O tensor precision (matches compute precision)
    dtype_map = _get_trt_dtype_map()
    if precision not in dtype_map:
        raise ValueError(f"Unknown precision: {precision}")
    io_dtype = dtype_map[precision]

    if io_dtype != trt.float32:
        try:
            for i in range(network.num_inputs):
                network.get_input(i).dtype = io_dtype
            for i in range(network.num_outputs):
                network.get_output(i).dtype = io_dtype
        except AttributeError:
            print("  TensorRT 11 uses model-declared I/O types; preserving ONNX tensor dtypes")

    print("  Building engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(
        f"  Engine saved: {engine_path} ({Path(engine_path).stat().st_size / 1024 / 1024:.1f} MB)",
    )

    # Verify the built engine has correct I/O types.
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    if engine is None:
        raise RuntimeError("Failed to deserialize the built TensorRT engine")
    print("  Verifying engine I/O:")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)
        dtype_str = _trt_dtype_str(dtype)
        print(f"    {name}: {dtype_str} ({mode})")
        if dtype != io_dtype:
            print(f"  WARNING: {name} is {dtype_str} but {_trt_dtype_str(io_dtype)} was requested!")


def height_to_shape(h: int, aspect: float = 16 / 9) -> tuple[int, int]:
    """Convert height to (width, height) assuming aspect ratio.

    Both width and height are aligned to 8 pixels, as required by many
    neural network architectures with pooling/striding layers.

    Args:
        h: Requested input height in pixels.
        aspect: Width divided by height before alignment.

    Returns:
        width: Aligned input width.
        height: Aligned input height.
    """
    # Align height to 8 first.
    h = (h + 7) // 8 * 8
    # Calculate width from aligned height.
    w = int(h * aspect)
    # Align width to 8.
    w = (w + 7) // 8 * 8
    return (w, h)


def main() -> int:
    """Export the NomosUni TensorRT engine.

    Returns:
        exit_code: Process exit status.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    model_name = "2x-nomosuni-compact"

    if flags.list:
        list_models()
        return 0

    min_h = flags.min_height or 720
    opt_h = flags.opt_height or 1080
    max_h = flags.max_height or 1080

    # Validate height constraints.
    if min_h > max_h:
        raise ValueError(f"--min-height ({min_h}) cannot be greater than --max-height ({max_h})")
    if opt_h < min_h or opt_h > max_h:
        raise ValueError(
            f"--opt-height ({opt_h}) must be between --min-height ({min_h}) and --max-height ({max_h})",
        )

    min_shape = height_to_shape(min_h)
    opt_shape = height_to_shape(opt_h)
    max_shape = height_to_shape(max_h)

    output_path = Path(flags.output or f"{model_name}_{opt_h}p_{flags.precision}.engine")
    if output_path.exists() and output_path.is_dir():
        raise ValueError(
            f"--output must include an engine filename, not a directory: {output_path}",
        )

    print("=" * 60)
    print("AI Upscale: TensorRT Engine Export")
    print("=" * 60)
    print(f"Model: {model_name}")
    print("  Fast universal upscale - compression, noise, and blur handling")
    print()

    model = load_model()

    if flags.onnx_only:
        # Save ONNX to current directory with sensible name.
        onnx_path = Path(f"{model_name}_{opt_h}p.onnx")
        cleanup_onnx = False
    else:
        # Temp file for intermediate ONNX.
        with tempfile.NamedTemporaryFile(
            prefix="export-tensorrt-",
            suffix=".onnx",
            delete=False,
        ) as tmp:
            onnx_path = Path(tmp.name)
        cleanup_onnx = True

    try:
        export_onnx(
            model,
            opt_shape,
            onnx_path,
            fixed_shape=min_shape == opt_shape == max_shape,
            precision=flags.precision,
        )

        if flags.onnx_only:
            print(f"\nONNX saved to: {onnx_path}")
            print("Skipping TensorRT build (--onnx-only). Build later with:")
            print(f"  trtexec --onnx={onnx_path} --saveEngine={output_path} --fp16")
            return 0

        build_engine(
            onnx_path,
            output_path,
            min_shape=min_shape,
            opt_shape=opt_shape,
            max_shape=max_shape,
            precision=flags.precision,
            workspace_gb=flags.workspace,
            opt_level=flags.opt_level,
        )
    finally:
        if cleanup_onnx and (onnx_file := Path(onnx_path)).exists():
            onnx_file.unlink()

    print()
    print("=" * 60)
    print("Export complete!")
    print("=" * 60)
    print()
    print(f"Model: {model_name} (2x upscale)")
    print(f"Engine accepts input heights from {min_h} to {max_h} (16:9)")
    print()
    print("Usage with FFmpeg:")
    print(
        f'  ffmpeg -i input.mp4 -vf "dnn_processing=dnn_backend=tensorrt:model={output_path}" output.mp4',
    )
    return 0


class Flags(Protocol):
    """Parsed command-line flags."""

    list: bool
    min_height: int | None
    opt_height: int | None
    max_height: int | None
    output: str | None
    precision: str
    workspace: int
    opt_level: int
    onnx_only: bool


class _NomosSRVGG(nn.Module):
    """SRVGG compact upscaler matching the Nomos checkpoint layout."""

    def __init__(self, *, num_conv: int, scale: int) -> None:
        super().__init__()
        self.scale = scale
        self.body = nn.ModuleList(
            [
                nn.Conv2d(3, 64, 3, 1, 1),
                nn.PReLU(num_parameters=64),
            ],
        )
        for _ in range(num_conv - 2):
            self.body.append(nn.Conv2d(64, 64, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=64))
        self.body.append(nn.Conv2d(64, 3 * scale * scale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(scale)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body[:-1]:
            out = layer(out)
        out = self.upsampler(self.body[-1](out))
        return out + F.interpolate(x, scale_factor=self.scale, mode="nearest")


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--list", "-l", action="store_true", help="Show the supported model")
    parser.add_argument(
        "--min-height",
        type=int,
        default=None,
        help="Minimum input height (default: 720)",
    )
    parser.add_argument(
        "--opt-height",
        type=int,
        default=None,
        help="Optimal input height (default: 1080)",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=None,
        help="Maximum input height (default: 1080)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output engine path",
    )
    parser.add_argument(
        "--precision",
        "-p",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Model precision for compute and I/O tensors (default: fp16)",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=8,
        help="TensorRT workspace size in GB (default: 8)",
    )
    parser.add_argument(
        "--opt-level",
        type=int,
        default=3,
        choices=[0, 1, 2, 3, 4, 5],
        help="TensorRT builder optimization level 0-5 (default: 3). Higher = more memory, potentially faster.",
    )
    parser.add_argument(
        "--onnx-only",
        action="store_true",
        help="Only export ONNX, skip TensorRT engine build",
    )


def _get_trt_dtype_map() -> dict[str, trt.DataType]:
    """Get mapping from precision string to TensorRT DataType."""

    dtype_map: dict[str, trt.DataType] = {
        "fp32": trt.float32,
        "fp16": trt.float16,
    }
    if hasattr(trt, "bfloat16"):
        dtype_map["bf16"] = trt.bfloat16
    return dtype_map


def _trt_dtype_str(dtype: trt.DataType) -> str:
    """Convert TensorRT DataType to human-readable string."""
    for name, dt in _get_trt_dtype_map().items():
        if dtype == dt:
            return name.upper()
    return str(dtype)


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
