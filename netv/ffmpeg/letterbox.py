"""Black-bar detection: where the picture sits in its frame, and the crop that keeps it.

A transcode samples one decoded frame a second through ``cropdetect`` into
``DETECTION_FILE`` in its output directory. Frames are scaled to a fixed grid first,
so the detected box is a fraction of the frame, whatever the source resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TypedDict

import math
import pathlib
import re


DETECTION_FILE: Final = "cropdetect.txt"
# Frames are sampled on this square grid, so detections read as fractions of the frame.
_GRID: Final = 1000


class CachedCrop(TypedDict):
    left: float
    top: float
    width: float
    height: float


@dataclass(kw_only=True, slots=True, frozen=True)
class CropBox:
    """The picture's part of a frame, as fractions of the frame's width and height."""

    left: float = 0.0
    top: float = 0.0
    width: float = 1.0
    height: float = 1.0

    @property
    def is_full(self) -> bool:
        return self == CropBox()

    def union(self, other: CropBox) -> CropBox:
        """The smallest box holding both: every picture either one saw.

        Args:
            other: Another picture area of the same frame.

        Returns:
            union: The box spanning both.
        """
        return _box(
            min(self.left, other.left),
            min(self.top, other.top),
            max(self.left + self.width, other.left + other.width),
            max(self.top + self.height, other.top + other.height),
        )

    def within(self, outer: CropBox) -> CropBox:
        """This box as fractions of ``outer``, clamped to it and with slivers removed.

        Args:
            outer: The part of the frame this box is reframed against, such as a
                crop the encoder already applied.

        Returns:
            box: The picture area of ``outer``'s frame.
        """
        return _snapped(
            (self.left - outer.left) / outer.width,
            (self.top - outer.top) / outer.height,
            (self.left + self.width - outer.left) / outer.width,
            (self.top + self.height - outer.top) / outer.height,
        )

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """The box in a ``width`` x ``height`` frame, rounded outward to even pixels.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.

        Returns:
            crop_width: Box width.
            crop_height: Box height.
            x: Left edge.
            y: Top edge.
        """
        x, right = _even_span(self.left, self.width, width)
        y, bottom = _even_span(self.top, self.height, height)
        return right - x, bottom - y, x, y

    def to_cache(self) -> CachedCrop:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    @classmethod
    def from_cache(cls, cached: CachedCrop) -> CropBox:
        return cls(
            left=cached["left"],
            top=cached["top"],
            width=cached["width"],
            height=cached["height"],
        )


def detection_args(output_dir: str, hw_frames: str, *, is_10bit: bool) -> list[str]:
    """An extra ffmpeg output sampling decoded frames for black bars.

    Args:
        output_dir: Directory receiving ``DETECTION_FILE``.
        hw_frames: ``-hwaccel_output_format`` of the decoded frames; "" for system memory.
        is_10bit: Whether VAAPI or QSV surfaces hold 10-bit frames.

    Returns:
        args: Output options to place before the command's main output.
    """
    # Cropdetect reads pixels in system memory; ffmpeg has no GPU version of it.
    download = {
        "cuda": (
            f"scale_cuda={_GRID}:{_GRID}:interp_algo=nearest:format=nv12,hwdownload,format=nv12,"
        ),
        "": "",
    }.get(hw_frames, f"hwdownload,format={'p010le' if is_10bit else 'nv12'},")
    scale = "" if hw_frames == "cuda" else f"scale={_GRID}:{_GRID}:flags=neighbor,"
    path = _filter_value(f"{output_dir}/{DETECTION_FILE}")
    # Cropdetect never resets, so each sample holds every picture edge seen so far.
    # direct=1: ffmpeg otherwise buffers ~32 KB (minutes of samples) before writing, and a
    # live transcode never exits to flush it.
    chain = f"fps=1,{download}{scale}cropdetect=skip=0,metadata=mode=print:direct=1:file={path}"
    # -filter:v, not -vf: callers find the main output's filters by "-vf".
    return ["-map", "0:v:0", "-filter:v", chain, "-f", "null", "-"]


def detected_crop(
    directory: str | pathlib.Path,
    *,
    min_samples: int = 60,
) -> CropBox | None:
    """The picture area a transcode's detection found, once it has sampled enough.

    Args:
        directory: Transcode output directory holding ``DETECTION_FILE``.
        min_samples: Samples (seconds of video) needed before the box is trusted;
            dark opening scenes understate the picture.

    Returns:
        crop: Every picture edge seen since the transcode started, or None before
            enough samples, for an all-black picture, or without readable detections.
    """
    try:
        text = (pathlib.Path(directory) / DETECTION_FILE).read_text(errors="replace")
    except OSError:
        return None
    # Ffmpeg truncates the file on start and may be mid-write: use the last whole sample.
    for block in reversed(text.split("frame:")[1:]):
        values = dict(re.findall(r"lavfi\.cropdetect\.(x1|x2|y1|y2)=(\d+)", block))
        frame = re.match(r"(\d+)", block)
        if len(values) < 4 or frame is None:
            continue
        x1, x2, y1, y2 = (int(values[key]) for key in ("x1", "x2", "y1", "y2"))
        if int(frame[1]) + 1 < min_samples or x1 > x2 or y1 > y2:
            return None
        return _snapped(x1 / _GRID, y1 / _GRID, (x2 + 1) / _GRID, (y2 + 1) / _GRID)
    return None


def _box(left: float, top: float, right: float, bottom: float) -> CropBox:
    # Rounded so boxes compare and serialize without float noise.
    return CropBox(
        left=round(left, 4),
        top=round(top, 4),
        width=round(right - left, 4),
        height=round(bottom - top, 4),
    )


def _snapped(
    left: float,
    top: float,
    right: float,
    bottom: float,
    *,
    min_bar: float = 0.01,
) -> CropBox:
    """A box clamped to the frame, ignoring bars too thin to be letterboxing."""
    left, top = max(left, 0.0), max(top, 0.0)
    right, bottom = min(right, 1.0), min(bottom, 1.0)
    return _box(
        0.0 if left < min_bar else left,
        0.0 if top < min_bar else top,
        1.0 if right > 1 - min_bar else right,
        1.0 if bottom > 1 - min_bar else bottom,
    )


def _even_span(start: float, length: float, size: int) -> tuple[int, int]:
    """Pixel edges of a fractional span, widened to even pixels (4:2:0 chroma)."""
    first = math.floor(start * size / 2) * 2
    last = min(size, math.ceil((start + length) * size / 2) * 2)
    return first, last


def _filter_value(value: str) -> str:
    """Quote ``value`` for an option of one filter in an ffmpeg filtergraph."""
    quoted = "'" + value.replace("'", "'\\''") + "'"
    return "".join(f"\\{c}" if c in "\\'[],;" else c for c in quoted)
