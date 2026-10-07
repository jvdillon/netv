#!/usr/bin/env python3
"""Benchmark the fixed 1080p NomosUni TensorRT engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import argparse
import json
import shutil
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = Path.home() / "ffmpeg_build/models"


def find_binary(name: str) -> Path:
    bundled = ROOT / "bin" / name
    if bundled.is_file():
        return bundled
    resolved = shutil.which(name)
    if resolved:
        return Path(resolved)
    raise FileNotFoundError(f"Unable to find {name}; add it to PATH or place it in {ROOT / 'bin'}")


@dataclass(frozen=True)
class Result:
    model: str
    elapsed_seconds: float
    startup_seconds: float
    source_seconds: float
    source_fps: float

    @property
    def speed(self) -> float:
        return self.source_seconds / self.elapsed_seconds

    @property
    def throughput_fps(self) -> float:
        return self.source_fps * self.speed


def probe_sample(ffprobe: Path, sample: Path) -> tuple[float, float]:
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration:stream=avg_frame_rate,width,height",
        "-of",
        "json",
        str(sample),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    if (stream["width"], stream["height"]) != (1920, 1080):
        raise RuntimeError(
            f"Benchmark sample must be 1920x1080, got {stream['width']}x{stream['height']}"
        )
    numerator, denominator = stream["avg_frame_rate"].split("/", 1)
    return float(payload["format"]["duration"]), float(numerator) / float(denominator)


def command_for(
    ffmpeg: Path,
    sample: Path,
    engine: Path,
    output_dir: Path,
    duration: float | None,
) -> list[str]:
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-init_hw_device",
        "cuda=cu",
        "-filter_hw_device",
        "cu",
        "-filter_threads",
        "4",
        "-i",
        str(sample),
    ]
    if duration is not None:
        command.extend(["-t", str(duration)])
    command.extend(
        [
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            (
                "format=rgb24,hwupload,"
                f"dnn_processing=dnn_backend=tensorrt:model={engine},"
                "scale_cuda=w=-2:h=2160"
            ),
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p5",
            "-rc",
            "constqp",
            "-qp",
            "20",
            "-bf",
            "0",
            "-spatial-aq",
            "1",
            "-temporal-aq",
            "1",
            "-g",
            "60",
            "-f",
            "hls",
            "-hls_time",
            "2",
            "-hls_list_size",
            "0",
            "-hls_segment_filename",
            str(output_dir / "seg%05d.ts"),
            str(output_dir / "stream.m3u8"),
        ]
    )
    return command


def run_once(
    ffmpeg: Path,
    sample: Path,
    engine: Path,
    source_seconds: float,
    source_fps: float,
    duration: float | None = None,
) -> Result:
    with tempfile.TemporaryDirectory(prefix="netv_nomos_benchmark_") as temp_dir:
        output_dir = Path(temp_dir)
        started = time.monotonic()
        startup_at: float | None = None
        process = subprocess.Popen(
            command_for(ffmpeg, sample, engine, output_dir, duration),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        while process.poll() is None:
            playlist = output_dir / "stream.m3u8"
            if startup_at is None and playlist.exists() and ".ts" in playlist.read_text():
                startup_at = time.monotonic()
            time.sleep(0.02)
        elapsed = time.monotonic() - started
        stderr = process.stderr.read() if process.stderr else ""
        if process.returncode != 0:
            detail = "\n".join(stderr.strip().splitlines()[-20:])
            raise RuntimeError(f"{engine.stem} failed:\n{detail}")
        measured_seconds = duration if duration is not None else source_seconds
        return Result(
            model=engine.stem,
            elapsed_seconds=elapsed,
            startup_seconds=(startup_at or time.monotonic()) - started,
            source_seconds=measured_seconds,
            source_fps=source_fps,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        type=Path,
        required=True,
        help="30-second 1920x1080 test clip",
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--warmup-seconds", type=float, default=5.0)
    parser.add_argument("--target-fps", type=float, default=60.0)
    args = parser.parse_args()

    ffmpeg = find_binary("ffmpeg")
    ffprobe = find_binary("ffprobe")
    source_seconds, source_fps = probe_sample(ffprobe, args.sample)
    engine = args.model_dir / "2x-nomosuni-compact_1080p_fp16.engine"
    for path in [args.sample, engine]:
        if not path.exists():
            raise FileNotFoundError(path)

    print(
        f"Sample: {args.sample.name} (1920x1080, {source_fps:.3f} FPS, "
        f"{source_seconds:.2f}s)"
    )
    print(f"Target: >= {args.target_fps:.3f} FPS end-to-end")
    print()

    print(f"Warming {engine.stem} for {args.warmup_seconds:.1f}s...")
    run_once(
        ffmpeg,
        args.sample,
        engine,
        source_seconds,
        source_fps,
        duration=args.warmup_seconds,
    )
    print(f"Measuring {engine.stem}...")
    result = run_once(ffmpeg, args.sample, engine, source_seconds, source_fps)
    status = "PASS" if result.throughput_fps >= args.target_fps else "FAIL"
    print(
        f"  {result.throughput_fps:.2f} FPS ({result.speed:.3f}x), "
        f"startup {result.startup_seconds:.3f}s: {status}"
    )


if __name__ == "__main__":
    main()
