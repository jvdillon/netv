"""Tests for the NomosUni TensorRT export script with faked ML libraries."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from importlib import util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Final
from urllib import request

import runpy
import sys

import pytest


_CWD: Final = Path(__file__).resolve().parent

SCRIPT = _CWD / "export-tensorrt.py"
CHECKPOINT = "2xNomosUni_compact_otf_medium.safetensors"


class _Param:
    def __init__(self, n: int) -> None:
        self.n = n

    def numel(self) -> int:
        return self.n


class _Module:
    forward: Callable[..., Any]

    def __init__(self) -> None:
        self.training = True
        self.params: list[_Param] = []

    def __call__(self, *args: Any) -> Any:
        return self.forward(*args)

    def children(self) -> Iterator[_Module]:
        for value in vars(self).values():
            if isinstance(value, _Module):
                yield value

    def parameters(self) -> Iterator[_Param]:
        yield from self.params
        for child in self.children():
            yield from child.parameters()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.loaded = state_dict

    def eval(self) -> _Module:
        self.training = False
        return self


class _ModuleList(_Module):
    def __init__(self, items: list[_Module]) -> None:
        super().__init__()
        self.items = list(items)

    def append(self, item: _Module) -> None:
        self.items.append(item)

    def __getitem__(self, index: Any) -> Any:
        return self.items[index]

    def children(self) -> Iterator[_Module]:
        yield from self.items


class _Conv2d(_Module):
    def __init__(self, cin: int, cout: int, k: int, stride: int, pad: int) -> None:
        super().__init__()
        assert (k, stride, pad) == (3, 1, 1)
        self.name = f"conv{cin}->{cout}"
        self.params = [_Param(cout * cin * k * k), _Param(cout)]

    def forward(self, x: list[Any]) -> list[Any]:
        return [*x, self.name]


class _PReLU(_Module):
    def __init__(self, num_parameters: int) -> None:
        super().__init__()
        self.params = [_Param(num_parameters)]

    def forward(self, x: list[Any]) -> list[Any]:
        return [*x, "prelu"]


class _PixelShuffle(_Module):
    def __init__(self, scale: int) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: list[Any]) -> list[Any]:
        return [*x, f"shuffle{self.scale}"]


def _interpolate(x: list[Any], scale_factor: int, mode: str) -> list[Any]:
    return [("interp", tuple(x), scale_factor, mode)]


class Recorder:
    def __init__(self) -> None:
        self.exports: list[dict[str, Any]] = []
        self.saved: list[tuple[Any, Any]] = []

    def export(self, model: Any, args: tuple[Any, ...], path: Any, **kwargs: Any) -> None:
        self.exports.append({"model": model, "args": args, "path": path, **kwargs})
        Path(path).write_bytes(b"onnx-bytes")

    def save(self, model: Any, path: Any) -> None:
        self.saved.append((model, path))


def _module(name: str, **attrs: Any) -> ModuleType:
    module = ModuleType(name)
    module.__dict__.update(attrs)
    return module


def _fake_modules(rec: Recorder) -> dict[str, ModuleType]:
    functional = _module("torch.nn.functional", interpolate=_interpolate)
    nn = _module(
        "torch.nn",
        Module=_Module,
        ModuleList=_ModuleList,
        Conv2d=_Conv2d,
        PReLU=_PReLU,
        PixelShuffle=_PixelShuffle,
        functional=functional,
    )
    onnx_export = _module("torch.onnx", export=rec.export)
    torch = _module(
        "torch",
        nn=nn,
        onnx=onnx_export,
        Tensor=object,
        randn=lambda *shape, device: ("randn", shape, device),
    )
    float16 = SimpleNamespace(
        convert_float_to_float16=lambda model, keep_io_types: ("fp16", model, keep_io_types),
    )
    safetensors_torch = _module("safetensors.torch", load_file=lambda path, device: {})
    return {
        "torch": torch,
        "torch.nn": nn,
        "torch.nn.functional": functional,
        "torch.onnx": onnx_export,
        "onnx": _module("onnx", load=lambda path: ("onnx", str(path)), save=rec.save),
        "onnxconverter_common": _module("onnxconverter_common", float16=float16),
        "safetensors": _module("safetensors", torch=safetensors_torch),
        "safetensors.torch": safetensors_torch,
        "tensorrt": make_trt(),
    }


class _Tensor:
    def __init__(self, name: str, *, settable: bool = True) -> None:
        self.name = name
        self.settable = settable
        self._dtype = "f32"

    @property
    def dtype(self) -> str:
        return self._dtype

    @dtype.setter
    def dtype(self, value: str) -> None:
        if not self.settable:
            raise AttributeError("read-only")
        self._dtype = value


def make_trt(
    *,
    explicit_batch: int | None = 0,
    fp16_flag: bool = True,
    bf16_flag: bool = True,
    bfloat16: bool = True,
    fast_fp16: bool | None = True,
    parse_errors: tuple[str, ...] = (),
    serialized: bytes | None = b"engine-bytes",
    engine: bool = True,
    io_settable: bool = True,
    engine_dtypes: tuple[str, str] | None = None,
) -> ModuleType:
    rec: dict[str, Any] = {"flags": [], "profiles": []}
    network = SimpleNamespace(
        inputs=[_Tensor("input", settable=io_settable)],
        outputs=[_Tensor("output", settable=io_settable)],
    )
    network.num_inputs = 1
    network.num_outputs = 1
    network.get_input = lambda i: network.inputs[i]
    network.get_output = lambda i: network.outputs[i]

    class Profile:
        def set_shape(self, name: str, **shapes: tuple[int, ...]) -> None:
            rec["profiles"].append((name, shapes))

    class Config:
        builder_optimization_level = 3

        def set_memory_pool_limit(self, pool: str, size: int) -> None:
            rec["pool"] = (pool, size)

        def add_optimization_profile(self, profile: Profile) -> None:
            rec["config_profile"] = profile

        def set_flag(self, flag: str) -> None:
            rec["flags"].append(flag)

    class Builder:
        def __init__(self, logger: object) -> None:
            rec["logger"] = logger
            if fast_fp16 is not None:
                self.platform_has_fast_fp16 = fast_fp16

        def create_network(self, flags: int) -> SimpleNamespace:
            rec["network_flags"] = flags
            return network

        def create_builder_config(self) -> Config:
            rec["config"] = Config()
            return rec["config"]

        def create_optimization_profile(self) -> Profile:
            return Profile()

        def build_serialized_network(self, net: object, config: Config) -> bytes | None:
            assert net is network
            assert config is rec["config"]
            return serialized

    class OnnxParser:
        def __init__(self, net: object, logger: object) -> None:
            self.num_errors = len(parse_errors)

        def parse(self, data: bytes) -> bool:
            rec["parsed"] = data
            return not parse_errors

        def get_error(self, i: int) -> str:
            return parse_errors[i]

    class Engine:
        num_io_tensors = 2

        def get_tensor_name(self, i: int) -> str:
            return ("input", "output")[i]

        def get_tensor_dtype(self, name: str) -> str:
            if engine_dtypes is not None:
                return engine_dtypes[name == "output"]
            return network.inputs[0].dtype if name == "input" else network.outputs[0].dtype

        def get_tensor_mode(self, name: str) -> str:
            return "IN" if name == "input" else "OUT"

    class Runtime:
        def __init__(self, logger: object) -> None:
            pass

        def deserialize_cuda_engine(self, data: bytes) -> Engine | None:
            assert data == serialized
            return Engine() if engine else None

    builder_flag = SimpleNamespace()
    if fp16_flag:
        builder_flag.FP16 = "FP16"
    if bf16_flag:
        builder_flag.BF16 = "BF16"
    creation_flag = SimpleNamespace()
    if explicit_batch is not None:
        creation_flag.EXPLICIT_BATCH = explicit_batch
    attrs: dict[str, Any] = {
        "rec": rec,
        "network": network,
        "Builder": Builder,
        "OnnxParser": OnnxParser,
        "Runtime": Runtime,
        "NetworkDefinitionCreationFlag": creation_flag,
        "MemoryPoolType": SimpleNamespace(WORKSPACE="WORKSPACE"),
        "BuilderFlag": builder_flag,
        "float32": "f32",
        "float16": "f16",
    }
    attrs["Logger"] = type("Logger", (), {"INFO": "INFO", "__init__": lambda self, lvl: None})
    if bfloat16:
        attrs["bfloat16"] = "bf16"
    return _module("tensorrt", **attrs)


@pytest.fixture
def rec() -> Recorder:
    return Recorder()


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch, rec: Recorder) -> dict[str, ModuleType]:
    modules = _fake_modules(rec)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return modules


@pytest.fixture
def exp(fakes: dict[str, ModuleType], monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    del fakes
    spec = util.spec_from_file_location("export_tensorrt", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def test_download_model_cached(exp: ModuleType, tmp_path: Path) -> None:
    (tmp_path / CHECKPOINT).write_bytes(b"w")
    assert exp.download_model(tmp_path) == tmp_path / CHECKPOINT


def test_download_model_fetches(
    exp: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def urlopen(url: str, timeout: int) -> FakeResponse:
        assert url.startswith("https://huggingface.co/Phips/")
        assert url.endswith(CHECKPOINT)
        assert timeout == 300
        return FakeResponse(b"x" * 1024 * 1024)

    monkeypatch.setattr(request, "urlopen", urlopen)
    assert exp.download_model(tmp_path) == tmp_path / CHECKPOINT
    assert (tmp_path / CHECKPOINT).read_bytes() == b"x" * 1024 * 1024
    assert "(1.0 MB)" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == [tmp_path / CHECKPOINT]


def test_download_model_empty_cleans_up(
    exp: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(request, "urlopen", lambda url, timeout: FakeResponse(b""))
    with pytest.raises(RuntimeError, match="Failed to download model .*Downloaded file is empty"):
        exp.download_model(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_download_model_network_error(
    exp: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def urlopen(url: str, timeout: int) -> FakeResponse:
        raise OSError("offline")

    monkeypatch.setattr(request, "urlopen", urlopen)
    with pytest.raises(RuntimeError, match="offline") as exc:
        exp.download_model(tmp_path)
    assert isinstance(exc.value.__cause__, OSError)
    assert list(tmp_path.iterdir()) == []


def test_list_models(exp: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    exp.list_models()
    assert "2x-nomosuni-compact      Fast universal upscale" in capsys.readouterr().out


def _state_dict() -> dict[str, SimpleNamespace]:
    return {
        "body.0.weight": SimpleNamespace(shape=(64, 3, 3, 3)),
        "body.0.bias": SimpleNamespace(shape=(64,)),
        "body.1.weight": SimpleNamespace(shape=(64,)),
        "body.2.weight": SimpleNamespace(shape=(64, 64, 3, 3)),
        "body.4.weight": SimpleNamespace(shape=(12, 64, 3, 3)),
    }


def test_build_nomos_model_forward(exp: ModuleType) -> None:
    model = exp.build_nomos_model(_state_dict())
    assert model.scale == 2
    assert model(["x"]) == [
        "x",
        "conv3->64",
        "prelu",
        "conv64->64",
        "prelu",
        "conv64->12",
        "shuffle2",
        ("interp", ("x",), 2, "nearest"),
    ]


def test_load_model(
    exp: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cache = tmp_path / "cache"
    state = _state_dict()
    loads: list[tuple[Path, str]] = []

    def load_file(path: Path, device: str) -> dict[str, SimpleNamespace]:
        loads.append((path, device))
        return state

    monkeypatch.setattr(exp, "load_file", load_file)
    monkeypatch.setattr(exp, "download_model", lambda d: d / CHECKPOINT)
    model = exp.load_model(cache)
    assert cache.is_dir()
    assert loads == [(cache / CHECKPOINT, "cpu")]
    assert model.loaded is state
    assert model.training is False
    # 1792 + 64 + 36928 + 64 + 6924 parameters.
    assert "Loaded SRVGGNetCompact (0.05M params), Scale: 2x" in capsys.readouterr().out


def test_load_model_default_cache(exp: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[Path] = []
    monkeypatch.setattr(Path, "mkdir", lambda self, parents, exist_ok: made.append(self))
    monkeypatch.setattr(exp, "download_model", lambda d: d / CHECKPOINT)
    monkeypatch.setattr(exp, "load_file", lambda path, device: _state_dict())
    exp.load_model()
    assert made == [Path("/opt/scratch/caches/netv/ai_upscale")]


def test_export_onnx_dynamic_fp32(
    exp: ModuleType,
    rec: Recorder,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exp.export_onnx("model", (64, 32), tmp_path / "m.onnx")
    (call,) = rec.exports
    assert call["args"] == (("randn", (1, 3, 32, 64), "cpu"),)
    assert call["opset_version"] == 17
    assert call["dynamic_axes"]["input"] == {2: "height", 3: "width"}
    assert rec.saved == []
    assert "ONNX export complete (dynamic H/W, fp32)" in capsys.readouterr().out


def test_export_onnx_fixed_fp16_and_bf16(
    exp: ModuleType,
    rec: Recorder,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "m.onnx"
    exp.export_onnx("model", (8, 8), path, fixed_shape=True, precision="fp16")
    assert rec.exports[0]["dynamic_axes"] is None
    assert rec.saved == [(("fp16", ("onnx", str(path)), False), path)]
    exp.export_onnx("model", (8, 8), path, fixed_shape=True, precision="bf16")
    out = capsys.readouterr().out
    assert "BF16 ONNX conversion is unavailable" in out
    assert "ONNX export complete (fixed shape, bf16)" in out
    assert len(rec.saved) == 1


def _build(exp: ModuleType, tmp_path: Path, precision: str = "fp16") -> Path:
    onnx_path = tmp_path / "m.onnx"
    onnx_path.write_bytes(b"onnx-bytes")
    engine_path = tmp_path / "m.engine"
    exp.build_engine(
        onnx_path,
        engine_path,
        (1280, 720),
        (1920, 1080),
        (3840, 2160),
        precision=precision,
        workspace_gb=2,
        opt_level=5,
    )
    return engine_path


def test_build_engine_fp16(
    exp: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trt = make_trt(engine_dtypes=("f16", "f32"))
    monkeypatch.setattr(exp, "trt", trt)
    engine_path = _build(exp, tmp_path)
    rec = trt.rec
    assert engine_path.read_bytes() == b"engine-bytes"
    assert rec["parsed"] == b"onnx-bytes"
    assert rec["network_flags"] == 1
    assert rec["pool"] == ("WORKSPACE", 2 << 30)
    assert rec["config"].builder_optimization_level == 5
    assert rec["profiles"] == [
        (
            "input",
            {"min": (1, 3, 720, 1280), "opt": (1, 3, 1080, 1920), "max": (1, 3, 2160, 3840)},
        ),
    ]
    assert rec["flags"] == ["FP16"]
    assert [t.dtype for t in [*trt.network.inputs, *trt.network.outputs]] == ["f16", "f16"]
    out = capsys.readouterr().out
    assert "    input: FP16 (IN)" in out
    assert "WARNING: output is FP32 but FP16 was requested!" in out


def test_build_engine_bf16(
    exp: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trt = make_trt()
    monkeypatch.setattr(exp, "trt", trt)
    _build(exp, tmp_path, precision="bf16")
    assert trt.rec["flags"] == ["FP16", "BF16"]
    assert trt.network.inputs[0].dtype == "bf16"


def test_build_engine_bf16_falls_back_to_fp16(
    exp: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trt = make_trt(bf16_flag=False)
    monkeypatch.setattr(exp, "trt", trt)
    _build(exp, tmp_path, precision="bf16")
    assert trt.rec["flags"] == ["FP16"]
    assert trt.network.outputs[0].dtype == "f16"
    assert "BF16 not supported by TensorRT, using FP16" in capsys.readouterr().out


def test_build_engine_tensorrt11_style(
    exp: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trt = make_trt(explicit_batch=None, fp16_flag=False, io_settable=False, fast_fp16=None)
    monkeypatch.setattr(exp, "trt", trt)
    _build(exp, tmp_path)
    assert trt.rec["network_flags"] == 0
    assert trt.rec["flags"] == []
    out = capsys.readouterr().out
    assert "preserving ONNX tensor dtypes" in out
    assert "WARNING: input is FP32 but FP16 was requested!" in out


def test_build_engine_slow_fp16_uses_fp32(
    exp: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trt = make_trt(fast_fp16=False)
    monkeypatch.setattr(exp, "trt", trt)
    _build(exp, tmp_path)
    assert trt.rec["flags"] == []
    assert trt.network.inputs[0].dtype == "f32"
    out = capsys.readouterr().out
    assert "FP16/BF16 not supported on this platform, using FP32" in out
    assert "WARNING" not in out.replace("Warning: FP16", "")


def test_build_engine_failures(
    exp: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(exp, "trt", make_trt(bfloat16=False))
    with pytest.raises(ValueError, match="Unknown precision: bf16"):
        _build(exp, tmp_path, precision="bf16")

    monkeypatch.setattr(exp, "trt", make_trt(parse_errors=("bad node", "bad edge")))
    with pytest.raises(RuntimeError, match="Failed to parse ONNX model"):
        _build(exp, tmp_path)
    out = capsys.readouterr().out
    assert "ONNX parse error: bad node" in out
    assert "ONNX parse error: bad edge" in out

    monkeypatch.setattr(exp, "trt", make_trt(serialized=None))
    with pytest.raises(RuntimeError, match="Failed to build TensorRT engine"):
        _build(exp, tmp_path)
    assert not (tmp_path / "m.engine").exists()

    monkeypatch.setattr(exp, "trt", make_trt(engine=False))
    with pytest.raises(RuntimeError, match="Failed to deserialize"):
        _build(exp, tmp_path)


def test_height_to_shape_and_dtype_names(exp: ModuleType) -> None:
    assert exp.height_to_shape(1080) == (1920, 1080)
    assert exp.height_to_shape(721) == (1296, 728)
    assert exp.height_to_shape(100, aspect=1.0) == (104, 104)
    assert exp._trt_dtype_str("f16") == "FP16"
    assert exp._trt_dtype_str("int8") == "int8"


def _main(exp: ModuleType, monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["export-tensorrt.py", *args])
    return exp.main()


def test_main_list(
    exp: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _main(exp, monkeypatch, "--list") == 0
    assert "Supported model" in capsys.readouterr().out


def test_main_validates_arguments(
    exp: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match=r"--min-height \(1080\) cannot be greater"):
        _main(exp, monkeypatch, "--min-height", "1080", "--max-height", "720")
    with pytest.raises(ValueError, match=r"--opt-height \(480\) must be between"):
        _main(exp, monkeypatch, "--opt-height", "480")
    with pytest.raises(ValueError, match=r"--opt-height \(2160\) must be between"):
        _main(exp, monkeypatch, "--opt-height", "2160")
    with pytest.raises(ValueError, match="must include an engine filename"):
        _main(exp, monkeypatch, "-o", str(tmp_path))


def test_main_onnx_only(
    exp: ModuleType,
    rec: Recorder,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(exp, "load_model", lambda: "model")
    assert _main(exp, monkeypatch, "--onnx-only", "-p", "fp32") == 0
    assert (tmp_path / "2x-nomosuni-compact_1080p.onnx").read_bytes() == b"onnx-bytes"
    assert rec.exports[0]["dynamic_axes"] is not None
    assert (
        "trtexec --onnx=2x-nomosuni-compact_1080p.onnx "
        "--saveEngine=2x-nomosuni-compact_1080p_fp32.engine --fp16"
    ) in capsys.readouterr().out


def test_main_builds_engine_and_removes_temp_onnx(
    exp: ModuleType,
    rec: Recorder,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trt = make_trt()
    monkeypatch.setattr(exp, "trt", trt)
    monkeypatch.setattr(exp, "load_model", lambda: "model")
    engine = tmp_path / "out.engine"
    argv = ["--min-height", "1080", "--max-height", "1080", "-o", str(engine), "--opt-level", "4"]
    assert _main(exp, monkeypatch, *argv, "--workspace", "1") == 0
    temp_onnx = Path(rec.exports[0]["path"])
    assert temp_onnx.name.startswith("export-tensorrt-")
    assert not temp_onnx.exists()
    assert rec.exports[0]["dynamic_axes"] is None
    assert engine.read_bytes() == b"engine-bytes"
    assert trt.rec["pool"] == ("WORKSPACE", 1 << 30)
    assert trt.rec["config"].builder_optimization_level == 4
    out = capsys.readouterr().out
    assert "Engine accepts input heights from 1080 to 1080 (16:9)" in out
    assert f"model={engine}" in out


def test_script_entrypoint(
    fakes: dict[str, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    del fakes
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--list"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert exc.value.code == 0
    assert "Supported model" in capsys.readouterr().out


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
