"""Tests for the NomosUni TensorRT export script with faked ML libraries."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast, override
from urllib import request

import argparse
import runpy
import sys

import pytest


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

    @override
    def children(self) -> Iterator[_Module]:
        yield from self.items


class _Conv2d(_Module):
    def __init__(self, cin: int, cout: int, k: int, stride: int, pad: int) -> None:
        super().__init__()
        assert (k, stride, pad) == (3, 1, 1)
        self.name = f"conv{cin}->{cout}"
        self.params = [_Param(cout * cin * k * k), _Param(cout)]

    @override
    def forward(self, x: list[Any]) -> list[Any]:
        return [*x, self.name]


class _PReLU(_Module):
    def __init__(self, num_parameters: int) -> None:
        super().__init__()
        self.params = [_Param(num_parameters)]

    @override
    def forward(self, x: list[Any]) -> list[Any]:
        return [*x, "prelu"]


class _PixelShuffle(_Module):
    def __init__(self, scale: int) -> None:
        super().__init__()
        self.scale = scale

    @override
    def forward(self, x: list[Any]) -> list[Any]:
        return [*x, f"shuffle{self.scale}"]


def _interpolate(x: list[Any], scale_factor: int, mode: str) -> list[Any]:
    return [("interp", tuple(x), scale_factor, mode)]


def _randn(*shape: int, device: str) -> tuple[str, tuple[int, ...], str]:
    return ("randn", shape, device)


def _convert_float_to_float16(model: object, keep_io_types: bool) -> tuple[str, object, bool]:
    return ("fp16", model, keep_io_types)


def _load_nothing(path: object, device: str) -> dict[str, object]:
    del path, device
    return {}


def _onnx_load(path: object) -> tuple[str, str]:
    return ("onnx", str(path))


def _cached_checkpoint(directory: Path) -> Path:
    return directory / CHECKPOINT


def _load_state(path: object, device: str) -> dict[str, SimpleNamespace]:
    del path, device
    return _state_dict()


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
        randn=_randn,
    )
    float16 = SimpleNamespace(convert_float_to_float16=_convert_float_to_float16)
    safetensors_torch = _module("safetensors.torch", load_file=_load_nothing)
    return {
        "torch": torch,
        "torch.nn": nn,
        "torch.nn.functional": functional,
        "torch.onnx": onnx_export,
        "onnx": _module("onnx", load=_onnx_load, save=rec.save),
        "onnxconverter_common": _module("onnxconverter_common", float16=float16),
        "safetensors": _module("safetensors", torch=safetensors_torch),
        "safetensors.torch": safetensors_torch,
    }


class FakeTrtexec:
    """Records ``trtexec`` argv and writes the engine it was asked to save."""

    def __init__(self, *, returncode: int = 0, write: bool = True) -> None:
        self.calls: list[list[str]] = []
        self.returncode = returncode
        self.write = write

    def run(self, cmd: list[str], **kwargs: object) -> SimpleNamespace:
        assert kwargs == {"capture_output": True, "text": True, "check": False}
        self.calls.append(cmd)
        if self.write:
            (save,) = [a.split("=", 1)[1] for a in cmd if a.startswith("--saveEngine=")]
            Path(save).write_bytes(b"engine-bytes")
        return SimpleNamespace(returncode=self.returncode, stdout="out\n", stderr="err\n")


def _fake_trtexec(
    exp: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    **kwargs: Any,
) -> FakeTrtexec:
    fake = FakeTrtexec(**kwargs)

    def find_trtexec() -> str | None:
        return "/bin/trtexec"

    monkeypatch.setattr(exp, "find_trtexec", find_trtexec)
    monkeypatch.setattr(exp.subprocess, "run", fake.run)
    return fake


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
    monkeypatch.delitem(sys.modules, "netv.upscale.export_tensorrt", raising=False)
    return import_module("netv.upscale.export_tensorrt")


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
    def urlopen(url: str, timeout: int) -> FakeResponse:
        del url, timeout
        return FakeResponse(b"")

    monkeypatch.setattr(request, "urlopen", urlopen)
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
    monkeypatch.setattr(exp, "download_model", _cached_checkpoint)
    model = exp.load_model(cache)
    assert cache.is_dir()
    assert loads == [(cache / CHECKPOINT, "cpu")]
    assert model.loaded is state
    assert model.training is False
    # 1792 + 64 + 36928 + 64 + 6924 parameters.
    assert "Loaded SRVGGNetCompact (0.05M params), Scale: 2x" in capsys.readouterr().out


def test_load_model_default_cache(exp: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[Path] = []

    def mkdir(self: Path, parents: bool, exist_ok: bool) -> None:
        del parents, exist_ok
        made.append(self)

    monkeypatch.setattr(Path, "mkdir", mkdir)
    monkeypatch.setattr(exp, "download_model", _cached_checkpoint)
    monkeypatch.setattr(exp, "load_file", _load_state)
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


def _build(
    exp: ModuleType,
    tmp_path: Path,
    precision: str = "fp16",
    *,
    shapes: tuple[tuple[int, int], ...] = ((1280, 720), (1920, 1080), (3840, 2160)),
) -> Path:
    onnx_path = tmp_path / "m.onnx"
    onnx_path.write_bytes(b"onnx-bytes")
    engine_path = tmp_path / "m.engine"
    exp.build_engine(
        onnx_path,
        engine_path,
        *shapes,
        precision=precision,
        workspace_gb=2,
        opt_level=5,
    )
    return engine_path


def test_build_engine_fp16_dynamic(
    exp: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = _fake_trtexec(exp, monkeypatch)
    engine_path = _build(exp, tmp_path)
    assert engine_path.read_bytes() == b"engine-bytes"
    assert fake.calls == [
        [
            "/bin/trtexec",
            f"--onnx={tmp_path / 'm.onnx'}",
            f"--saveEngine={engine_path}",
            "--memPoolSize=workspace:2048",
            "--builderOptimizationLevel=5",
            "--skipInference",
            "--minShapes=input:1x3x720x1280",
            "--optShapes=input:1x3x1080x1920",
            "--maxShapes=input:1x3x2160x3840",
            "--fp16",
            "--inputIOFormats=fp16:chw",
            "--outputIOFormats=fp16:chw",
        ],
    ]
    assert "Engine saved:" in capsys.readouterr().out


def test_build_engine_bf16_fixed_shape(
    exp: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_trtexec(exp, monkeypatch)
    _build(exp, tmp_path, precision="bf16", shapes=((1920, 1080),) * 3)
    (cmd,) = fake.calls
    assert not [a for a in cmd if "Shapes=" in a]
    assert cmd[-4:] == [
        "--fp16",
        "--bf16",
        "--inputIOFormats=bf16:chw",
        "--outputIOFormats=bf16:chw",
    ]


def test_build_engine_fp32_sets_no_precision_flags(
    exp: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_trtexec(exp, monkeypatch)
    _build(exp, tmp_path, precision="fp32")
    (cmd,) = fake.calls
    assert cmd[-1] == "--maxShapes=input:1x3x2160x3840"


def test_build_engine_failures(
    exp: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_trtexec(exp, monkeypatch, returncode=1)
    with pytest.raises(RuntimeError, match=r"trtexec failed \(1\):\nout\nerr"):
        _build(exp, tmp_path)

    (tmp_path / "m.engine").unlink()
    _fake_trtexec(exp, monkeypatch, write=False)
    with pytest.raises(RuntimeError, match=r"trtexec failed \(0\)"):
        _build(exp, tmp_path)

    def find_nothing() -> str | None:
        return None

    monkeypatch.setattr(exp, "find_trtexec", find_nothing)
    with pytest.raises(RuntimeError, match="trtexec not found"):
        _build(exp, tmp_path)


def test_find_trtexec(
    exp: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback = tmp_path / "trtexec"

    def which_found(name: str) -> str | None:
        return f"/path/{name}"

    def which_missing(name: str) -> str | None:
        del name
        return None

    monkeypatch.setattr(exp.shutil, "which", which_found)
    assert exp.find_trtexec(fallback) == "/path/trtexec"
    monkeypatch.setattr(exp.shutil, "which", which_missing)
    assert exp.find_trtexec(fallback) is None
    fallback.write_bytes(b"")
    assert exp.find_trtexec(fallback) == str(fallback)


def test_height_to_shape(exp: ModuleType) -> None:
    assert exp.height_to_shape(1080) == (1920, 1080)
    assert exp.height_to_shape(721) == (1296, 728)
    assert exp.height_to_shape(100, aspect=1.0) == (104, 104)


def _parser(exp: ModuleType) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(exit_on_error=False)
    exp._add_arguments(parser)
    return parser


def test_arguments_defaults_and_short_flags(exp: ModuleType) -> None:
    parser = _parser(exp)
    flags = parser.parse_args([])
    assert (flags.list, flags.output, flags.precision) == (False, None, "fp16")
    assert (flags.workspace, flags.opt_level, flags.onnx_only) == (8, 3, False)
    assert (flags.min_height, flags.opt_height, flags.max_height) == (None, None, None)
    flags = parser.parse_args(["-l", "-o", "e.engine", "-p", "bf16"])
    assert (flags.list, flags.output, flags.precision) == (True, "e.engine", "bf16")
    assert "Show the supported model" in parser.format_help()


@pytest.mark.parametrize("precision", ["fp16", "bf16", "fp32"])
def test_arguments_accept_each_precision(exp: ModuleType, precision: str) -> None:
    assert _parser(exp).parse_args(["-p", precision]).precision == precision


@pytest.mark.parametrize("level", [0, 1, 2, 3, 4, 5])
def test_arguments_accept_each_opt_level(exp: ModuleType, level: int) -> None:
    assert _parser(exp).parse_args(["--opt-level", str(level)]).opt_level == level


@pytest.mark.parametrize(("flag", "value"), [("-p", "int8"), ("--opt-level", "6")])
def test_arguments_reject_out_of_range(exp: ModuleType, flag: str, value: str) -> None:
    with pytest.raises(argparse.ArgumentError, match="invalid choice"):
        _parser(exp).parse_args([flag, value])


def _main(exp: ModuleType, monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["export-tensorrt.py", *args])
    # A module loaded from a file path exposes its attributes as Any.
    return cast(int, exp.main())


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
    fake = _fake_trtexec(exp, monkeypatch)
    monkeypatch.setattr(exp, "load_model", lambda: "model")
    engine = tmp_path / "out.engine"
    argv = ["--min-height", "1080", "--max-height", "1080", "-o", str(engine), "--opt-level", "4"]
    assert _main(exp, monkeypatch, *argv, "--workspace", "1") == 0
    temp_onnx = Path(rec.exports[0]["path"])
    assert temp_onnx.name.startswith("export-tensorrt-")
    assert not temp_onnx.exists()
    assert rec.exports[0]["dynamic_axes"] is None
    assert engine.read_bytes() == b"engine-bytes"
    (cmd,) = fake.calls
    assert "--memPoolSize=workspace:1024" in cmd
    assert "--builderOptimizationLevel=4" in cmd
    out = capsys.readouterr().out
    assert "Engine accepts input heights from 1080 to 1080 (16:9)" in out
    assert f"model={engine}" in out


def test_script_entrypoint(
    fakes: dict[str, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    del fakes
    monkeypatch.setattr(sys, "argv", ["export_tensorrt", "--list"])
    monkeypatch.delitem(sys.modules, "netv.upscale.export_tensorrt", raising=False)
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("netv.upscale.export_tensorrt", run_name="__main__")
    assert exc.value.code == 0
    assert "Supported model" in capsys.readouterr().out


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
