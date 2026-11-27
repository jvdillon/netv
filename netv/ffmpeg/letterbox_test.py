"""Tests for black-bar detection and crop boxes."""

from pathlib import Path

import subprocess

import pytest

from netv.ffmpeg.letterbox import DETECTION_FILE, CropBox, detected_crop, detection_args


_FULL = CropBox()
# A 2.40:1 picture letterboxed in 16:9: rows 140-939 of 1080; other tests share it.
SCOPE = CropBox(top=0.13, height=0.74)


def _block(frame: int, x1: int, x2: int, y1: int, y2: int) -> str:
    return (
        f"frame:{frame}    pts:{frame}       pts_time:{frame}\n"
        f"lavfi.cropdetect.x1={x1}\nlavfi.cropdetect.x2={x2}\n"
        f"lavfi.cropdetect.y1={y1}\nlavfi.cropdetect.y2={y2}\n"
        f"lavfi.cropdetect.w={x2 - x1 + 1}\nlavfi.cropdetect.h={y2 - y1 + 1}\n"
        "lavfi.cropdetect.limit=0.094118\n"
    )


def _detections(directory: Path, *blocks: str) -> Path:
    (directory / DETECTION_FILE).write_text("".join(blocks))
    return directory


class TestCropBox:
    def test_default_is_the_full_frame(self):
        assert _FULL.is_full
        assert not SCOPE.is_full

    def test_union_keeps_every_picture_seen(self):
        wide = CropBox(left=0.1, top=0.2, width=0.8, height=0.5)
        tall = CropBox(left=0.2, top=0.1, width=0.5, height=0.8)
        assert wide.union(tall) == CropBox(left=0.1, top=0.1, width=0.8, height=0.8)

    def test_within_reframes_against_an_applied_crop(self):
        applied = CropBox(top=0.1, height=0.8)
        assert CropBox(top=0.2, height=0.6).within(applied) == CropBox(top=0.125, height=0.75)

    def test_within_clamps_a_picture_larger_than_the_applied_crop(self):
        assert _FULL.within(SCOPE) == _FULL

    def test_within_ignores_slivers(self):
        """A bar under 1% is encoder noise, not letterboxing; cropping it shifts the frame."""
        applied = CropBox(top=0.13, height=0.74)
        assert CropBox(top=0.135, height=0.73).within(applied) == _FULL

    @pytest.mark.parametrize(
        ("box", "expected"),
        [
            (SCOPE, (1920, 800, 0, 140)),
            (_FULL, (1920, 1080, 0, 0)),
            # Odd edges round outward to even pixels: no picture is cut.
            (CropBox(left=0.1001, top=0.1001, width=0.5, height=0.5), (962, 542, 192, 108)),
        ],
    )
    def test_pixels_round_outward_to_even(self, box: CropBox, expected: tuple[int, int, int, int]):
        assert box.pixels(1920, 1080) == expected

    def test_cache_round_trip(self):
        assert CropBox.from_cache(SCOPE.to_cache()) == SCOPE
        assert SCOPE.to_cache() == {"left": 0.0, "top": 0.13, "width": 1.0, "height": 0.74}


class TestDetectedCrop:
    def test_reads_the_last_complete_sample(self, tmp_path: Path):
        directory = _detections(
            tmp_path,
            _block(58, 0, 999, 200, 799),
            _block(59, 0, 999, 130, 869),
            "frame:60    pts:60\nlavfi.cropdetect.x1=0\n",
        )
        assert detected_crop(directory) == SCOPE

    def test_needs_a_minute_of_samples(self, tmp_path: Path):
        """Early dark scenes understate the picture; a minute lets it reach its edges."""
        directory = _detections(tmp_path, _block(58, 0, 999, 130, 869))
        assert detected_crop(directory) is None
        assert detected_crop(directory, min_samples=59) == SCOPE

    @pytest.mark.parametrize(
        "edges",
        [(999, 0, 999, 0), (999, 0, 130, 869), (0, 999, 999, 0)],
        ids=["all-black", "no-columns", "no-rows"],
    )
    def test_no_picture_yet(self, tmp_path: Path, edges: tuple[int, int, int, int]):
        directory = _detections(tmp_path, _block(59, *edges))
        assert detected_crop(directory) is None

    def test_a_one_pixel_picture_is_still_a_picture(self, tmp_path: Path):
        directory = _detections(tmp_path, _block(59, 500, 500, 300, 300))
        assert detected_crop(directory) == CropBox(left=0.5, top=0.3, width=0.001, height=0.001)

    def test_undecodable_bytes_do_not_hide_a_sample(self, tmp_path: Path):
        (tmp_path / DETECTION_FILE).write_bytes(b"\xff\xfe" + _block(59, 0, 999, 130, 869).encode())
        assert detected_crop(tmp_path) == SCOPE

    @pytest.mark.parametrize("content", ["", "garbage\n", "frame:x\nlavfi.cropdetect.x1=a\n"])
    def test_unreadable_detections(self, tmp_path: Path, content: str):
        (tmp_path / DETECTION_FILE).write_text(content)
        assert detected_crop(tmp_path) is None

    def test_missing_file(self, tmp_path: Path):
        assert detected_crop(tmp_path) is None

    def test_slivers_snap_to_the_frame_edge(self, tmp_path: Path):
        directory = _detections(tmp_path, _block(59, 5, 994, 130, 869))
        assert detected_crop(directory) == SCOPE

    def test_pillarbox(self, tmp_path: Path):
        directory = _detections(tmp_path, _block(59, 125, 874, 0, 999))
        assert detected_crop(directory) == CropBox(left=0.125, width=0.75)


class TestDetectionArgs:
    @pytest.mark.parametrize(
        ("hw_frames", "is_10bit", "download"),
        [
            ("", False, "scale=1000:1000:flags=neighbor,"),
            (
                "cuda",
                True,
                "scale_cuda=1000:1000:interp_algo=nearest:format=nv12,hwdownload,format=nv12,",
            ),
            ("vaapi", False, "hwdownload,format=nv12,scale=1000:1000:flags=neighbor,"),
            ("qsv", True, "hwdownload,format=p010le,scale=1000:1000:flags=neighbor,"),
        ],
    )
    def test_samples_one_frame_a_second(self, hw_frames: str, is_10bit: bool, download: str):
        args = detection_args("/out", hw_frames, is_10bit=is_10bit)
        assert args == [
            "-map",
            "0:v:0",
            "-filter:v",
            (
                f"fps=1,{download}cropdetect=skip=0,"
                f"metadata=mode=print:direct=1:file=\\'/out/{DETECTION_FILE}\\'"
            ),
            "-f",
            "null",
            "-",
        ]

    def test_escapes_the_output_directory(self):
        chain = detection_args("/a:b,c [d]';e", "", is_10bit=False)[3]
        assert chain.endswith(
            "file=\\'/a:b\\,c \\[d\\]\\'\\\\\\'\\'\\;e/" + DETECTION_FILE + "\\'",
        )


@pytest.mark.cli_ffmpeg
def test_detections_reach_disk_while_ffmpeg_runs(tmp_path: Path):
    """A live transcode never exits; buffered detections would arrive minutes late."""
    source = "color=black:s=1920x1080:r=10,drawbox=x=0:y=140:w=1920:h=800:c=white:t=fill"
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-progress",
            "pipe:1",
            "-re",
            "-f",
            "lavfi",
            "-i",
            source,
            *detection_args(str(tmp_path), "", is_10bit=False),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        # Progress reports the samples written so far; ffmpeg keeps running.
        for line in process.stdout:
            if line.startswith("frame=") and int(line.removeprefix("frame=")) >= 2:
                break
        assert process.poll() is None
        assert detected_crop(tmp_path, min_samples=2) == SCOPE
    finally:
        process.kill()
        process.wait()


@pytest.mark.cli_ffmpeg
@pytest.mark.parametrize("bars", ["letterbox", "pillarbox"])
def test_ffmpeg_finds_the_bars(tmp_path: Path, bars: str):
    directory = tmp_path / "we:ird,dir [x]';y"
    directory.mkdir()
    box = "x=0:y=140:w=1920:h=800" if bars == "letterbox" else "x=240:y=0:w=1440:h=1080"
    source = tmp_path / "bars.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=black:s=1920x1080:r=10:d=4,drawbox={box}:c=white:t=fill",
            "-c:v",
            "libx264",
            str(source),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            str(source),
            *detection_args(str(directory), "", is_10bit=False),
        ],
        check=True,
    )
    expected = SCOPE if bars == "letterbox" else CropBox(left=0.125, width=0.75)
    assert detected_crop(directory, min_samples=4) == expected
    assert expected.pixels(1920, 1080) == (
        (1920, 800, 0, 140) if bars == "letterbox" else (1440, 1080, 240, 0)
    )


if __name__ == "__main__":
    from codedoctor.lib.testing.runner import test_main

    test_main(__file__)
