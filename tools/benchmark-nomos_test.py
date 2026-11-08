"""Tests for the NomosUni TensorRT benchmark driver."""

from __future__ import annotations

from collections.abc import Callable
from importlib import util
from pathlib import Path
from types import ModuleType
from typing import Final

import json
import runpy
import shutil
import subprocess
import sys
import time

import pytest


_CWD: Final = Path(__file__).resolve().parent

SCRIPT = _CWD / "benchmark-nomos.py"
ENGINE_NAME = "2x-nomosuni-compact_1080p_fp16.engine"


def _load() -> ModuleType:
    spec = util.spec_from_file_location("benchmark_nomos", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = util.module_from_spec(spec)
    # Dataclasses resolves string annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load()


def _probe_stdout(width: int = 1920, height: int = 1080) -> str:
    return json.dumps(
        {
            "streams": [{"width": width, "height": height, "avg_frame_rate": "30000/1001"}],
            "format": {"duration": "30.0"},
        },
    )


class FakeProcess:
    """Popen stand-in that writes the HLS playlist once, then exits."""

    def __init__(self, command: list[str], returncode: int, stderr: str | None) -> None:
        self.playlist = Path(command[-1])
        self.returncode = returncode
        self.polls = 0
        self.stderr = None if stderr is None else _Reader(stderr)

    def poll(self) -> int | None:
        self.polls += 1
        if self.polls == 1:
            self.playlist.write_text("#EXTM3U\n")
            return None
        if self.polls == 2:
            self.playlist.write_text("#EXTM3U\nseg00000.ts\n")
            return None
        return self.returncode


class _Reader:
    def __init__(self, text: str) -> None:
        self.text = text

    def read(self) -> str:
        return self.text


def _popen_factory(
    commands: list[list[str]],
    returncode: int = 0,
    stderr: str | None = "",
) -> Callable[..., FakeProcess]:
    def popen(command: list[str], **kwargs: object) -> FakeProcess:
        assert kwargs["stderr"] is subprocess.PIPE
        commands.append(command)
        return FakeProcess(command, returncode, stderr)

    return popen


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _: None)


def test_find_binary_prefers_bundled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench, "ROOT", tmp_path)
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "ffmpeg").write_text("")
    assert bench.find_binary("ffmpeg") == tmp_path / "bin" / "ffmpeg"

    monkeypatch.setattr(
        shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name == "ffprobe" else None,
    )
    assert bench.find_binary("ffprobe") == Path("/usr/bin/ffprobe")
    with pytest.raises(FileNotFoundError, match=r"Unable to find nope; .* place it in .*bin"):
        bench.find_binary("nope")


def test_result_rates() -> None:
    result = bench.Result(
        model="m",
        elapsed_seconds=10.0,
        startup_seconds=1.0,
        source_seconds=30.0,
        source_fps=24.0,
    )
    assert result.speed == 3.0
    assert result.throughput_fps == 72.0


def test_probe_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    stdout = _probe_stdout()

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs == {"capture_output": True, "text": True, "check": True}
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=stdout)

    monkeypatch.setattr(subprocess, "run", run)
    duration, fps = bench.probe_sample(Path("/p/ffprobe"), Path("clip.mp4"))
    assert duration == 30.0
    assert fps == pytest.approx(29.97, abs=1e-3)
    assert calls[0][0] == "/p/ffprobe"
    assert calls[0][-1] == "clip.mp4"

    stdout = _probe_stdout(1280, 720)
    with pytest.raises(RuntimeError, match="must be 1920x1080, got 1280x720"):
        bench.probe_sample(Path("ffprobe"), Path("clip.mp4"))


def test_ffmpeg_command_duration() -> None:
    command = bench.ffmpeg_command(Path("ff"), Path("s.mp4"), Path("e.engine"), Path("/o"), 2.5)
    assert command[:2] == ["ff", "-hide_banner"]
    assert command[command.index("-i") + 1 : command.index("-i") + 4] == ["s.mp4", "-t", "2.5"]
    assert "dnn_processing=dnn_backend=tensorrt:model=e.engine" in command[command.index("-vf") + 1]
    assert command[-2:] == ["/o/seg%05d.ts", "/o/stream.m3u8"]
    assert "-t" not in bench.ffmpeg_command(Path("ff"), Path("s"), Path("e"), Path("/o"), None)


def test_run_once_measures_startup(monkeypatch: pytest.MonkeyPatch, no_sleep: None) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(subprocess, "Popen", _popen_factory(commands, stderr=None))
    result = bench.run_once(Path("ff"), Path("s"), Path("/m/model.engine"), 30.0, 25.0)
    assert result.model == "model"
    assert result.source_seconds == 30.0
    assert result.source_fps == 25.0
    assert 0 <= result.startup_seconds <= result.elapsed_seconds
    assert "-t" not in commands[0]

    result = bench.run_once(Path("ff"), Path("s"), Path("m.engine"), 30.0, 25.0, duration=4.0)
    assert result.source_seconds == 4.0


def test_run_once_without_playlist_uses_end_time(monkeypatch: pytest.MonkeyPatch) -> None:
    class Done:
        returncode = 0
        stderr = None

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def poll(self) -> int:
            return 0

    monkeypatch.setattr(subprocess, "Popen", Done)
    result = bench.run_once(Path("ff"), Path("s"), Path("m.engine"), 30.0, 25.0)
    assert result.startup_seconds >= result.elapsed_seconds


def test_run_once_reports_failure_tail(monkeypatch: pytest.MonkeyPatch, no_sleep: None) -> None:
    stderr = "\n".join(f"line {i}" for i in range(30)) + "\n"
    monkeypatch.setattr(subprocess, "Popen", _popen_factory([], returncode=1, stderr=stderr))
    with pytest.raises(RuntimeError) as exc:
        bench.run_once(Path("ff"), Path("s"), Path("m.engine"), 30.0, 25.0)
    message = str(exc.value)
    assert message.startswith("m failed:\nline 10\n")
    assert message.endswith("line 29")


def _patch_tools(monkeypatch: pytest.MonkeyPatch, commands: list[list[str]]) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_: subprocess.CompletedProcess(command, 0, stdout=_probe_stdout()),
    )
    monkeypatch.setattr(subprocess, "Popen", _popen_factory(commands))


@pytest.mark.parametrize(("target", "status"), [("1", "PASS"), ("1e9", "FAIL")])
def test_main_warms_then_measures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_sleep: None,
    target: str,
    status: str,
) -> None:
    sample = tmp_path / "clip.mp4"
    sample.write_bytes(b"")
    (tmp_path / ENGINE_NAME).write_bytes(b"")
    monkeypatch.setattr(bench, "ROOT", tmp_path)
    commands: list[list[str]] = []
    _patch_tools(monkeypatch, commands)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark-nomos.py",
            "--sample",
            str(sample),
            "--model-dir",
            str(tmp_path),
            "--warmup-seconds",
            "2",
            "--target-fps",
            target,
        ],
    )
    assert bench.main() == 0
    out = capsys.readouterr().out
    assert "Sample: clip.mp4 (1920x1080, 29.970 FPS, 30.00s)" in out
    assert "Warming 2x-nomosuni-compact_1080p_fp16 for 2.0s..." in out
    assert out.rstrip().endswith(f": {status}")
    assert [c[0] for c in commands] == ["/usr/bin/ffmpeg", "/usr/bin/ffmpeg"]
    assert "2.0" in commands[0]
    assert "-t" not in commands[1]


def test_main_requires_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sample = tmp_path / "clip.mp4"
    sample.write_bytes(b"")
    monkeypatch.setattr(bench, "ROOT", tmp_path)
    _patch_tools(monkeypatch, [])
    argv = ["b", "--sample", str(sample), "--model-dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(FileNotFoundError, match=ENGINE_NAME):
        bench.main()


def test_script_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_sleep: None,
) -> None:
    sample = tmp_path / "clip.mp4"
    sample.write_bytes(b"")
    (tmp_path / ENGINE_NAME).write_bytes(b"")
    _patch_tools(monkeypatch, [])
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--sample", str(sample), "--model-dir", str(tmp_path)],
    )
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert exc.value.code == 0
    assert "Target: >= 60.000 FPS end-to-end" in capsys.readouterr().out


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
