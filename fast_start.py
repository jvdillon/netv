"""Single upstream ingest and independent local encoders for live fast start."""

from datetime import UTC, datetime

import pathlib
import re
import time

from ffmpeg_command import (
    PASSTHROUGH_AUDIO_CODECS,
    HwAccel,
    MediaInfo,
    build_hls_ffmpeg_cmd,
    get_live_hls_list_size,
    http_reconnect_args,
    limit_live_video_bitrate,
    live_video_bitrates,
)


_PLAYLIST_ENTRIES = re.compile(r"#EXTINF:([\d.]+),[^\n]*\n(?:#[^\n]*\n)*([^#\n]+)")


def ingest_command(
    url: str,
    directory: str,
    user_agent: str | None,
    *,
    restarting: bool = False,
) -> list[str]:
    # Remux only. No probe subprocess and no encoder may open the upstream URL.
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    cmd += http_reconnect_args(url, max_delay=5)
    if user_agent:
        cmd += ["-user_agent", user_agent]
    output = [
        "-probesize",
        "5000000",
        "-analyzeduration",
        "1000000",
        "-i",
        url,
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-c",
        "copy",
        "-f",
        "hls",
        "-hls_time",
        "1",
        "-hls_list_size",
        "120",
        "-hls_flags",
        "delete_segments+temp_file" + ("+discont_start" if restarting else ""),
    ]
    if restarting:
        output += ["-hls_start_number_source", "epoch_us"]
    output += [
        "-hls_segment_filename",
        f"{directory}/input_%06d.ts",
        f"{directory}/input.m3u8",
    ]
    return cmd + output


def encoder_command(
    directory: str,
    hw: HwAccel,
    resolution: str,
    quality: str,
    deinterlace: bool,
    high: bool,
    *,
    audio: MediaInfo | None = None,
    audio_passthrough: bool = False,
    restarting: bool = False,
) -> list[str]:
    cmd = build_hls_ffmpeg_cmd(
        f"{directory}/input.m3u8",
        hw,
        directory,
        False,
        [],
        None,
        resolution,
        quality,
        None,
        deinterlace,
        allow_upscale=high,
        audio_info=audio,
        audio_passthrough=audio_passthrough,
    )
    index = cmd.index("-i")
    cmd[index:index] = ["-live_start_index", "-3" if restarting else "0"]
    cmd[cmd.index("-probesize") + 1] = "500000"
    cmd[cmd.index("-analyzeduration") + 1] = "500000"
    # Preserve the shared source timeline through both independent encoders.
    cmd[1:1] = ["-copyts"]
    prefix = "high" if high else "low"
    cmd[cmd.index("-hls_segment_filename") + 1] = f"{directory}/{prefix}_%06d.ts"
    flags = "delete_segments+temp_file"
    if restarting:
        flags += "+append_list+discont_start"
    cmd[cmd.index("-hls_flags") + 1] = flags
    duration = "2"
    cmd[cmd.index("-hls_time") + 1] = duration
    cmd[cmd.index("-hls_list_size") + 1] = str(get_live_hls_list_size(float(duration)))
    cmd[-1:-1] = [
        "-force_key_frames",
        f"expr:if(isnan(prev_forced_t),1,gte(t,prev_forced_t+{duration}))",
    ]
    cmd[-1] = f"{directory}/{prefix}.m3u8"
    limit_live_video_bitrate(cmd, resolution)
    return cmd


def master_playlist(resolution: str, *, include_high: bool, audio_bitrate: int = 0) -> str:
    """Stable rendition URLs; clients keep high disabled until health permits it.

    audio_bitrate covers surround audio beyond the stereo AAC in the usual 10% margin.
    """
    renditions = [("low", "720p")]
    if include_high:
        renditions.append(("high", resolution))
    lines = ["#EXTM3U", "#EXT-X-VERSION:3"]
    for name, size in renditions:
        _, maximum = live_video_bitrates(size)
        # Include room for AAC and MPEG-TS overhead in the advertised bandwidth.
        bandwidth = int(maximum * 1.1) + audio_bitrate
        lines += [f"#EXT-X-STREAM-INF:BANDWIDTH={bandwidth}", f"{name}.m3u8"]
    return "\n".join(lines) + "\n"


def surround_audio_bitrate(audio: MediaInfo | None, passthrough: bool) -> int:
    """Extra bandwidth to advertise for surround output (Dolby copies at up to 640 kbps)."""
    if audio and passthrough and audio.audio_codec in PASSTHROUGH_AUDIO_CODECS:
        return 640_000
    if audio is None or audio.audio_channels > 2:
        return 384_000
    return 0


def playlist_duration(directory: str, name: str) -> float:
    try:
        content = (pathlib.Path(directory) / name).read_text()
        return sum(float(value) for value in re.findall(r"#EXTINF:([\d.]+)", content))
    except (OSError, ValueError):
        return 0


def segment_pts(path: pathlib.Path) -> float | None:
    """First video PES presentation timestamp in an MPEG-TS segment, in seconds."""
    try:
        with path.open("rb") as source:
            data = source.read(188 * 512)
    except OSError:
        return None
    for offset in range(0, len(data) - 187, 188):
        packet = data[offset : offset + 188]
        if packet[0] != 0x47 or not packet[1] & 0x40:
            continue
        control = (packet[3] >> 4) & 3
        if control not in (1, 3):
            continue
        start = 4 if control == 1 else 5 + packet[4]
        pes = packet[start:]
        if len(pes) < 14 or pes[:3] != b"\x00\x00\x01" or not 0xE0 <= pes[3] <= 0xEF:
            continue
        if not pes[7] & 0x80:
            continue
        p = pes[9:14]
        pts = (p[0] >> 1 & 7) << 30 | p[1] << 22 | (p[2] >> 1) << 15 | p[3] << 7 | p[4] >> 1
        return pts / 90000
    return None


def dated_playlist(directory: str, content: str, origin_pts: float, origin_time: float) -> str:
    """Map both renditions to the same clock, independent of encoder warm-up."""
    lines = content.splitlines()
    result = []
    reset_clock = False
    duration = 0.0
    for line in lines:
        if line.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
            continue
        if line.startswith("#EXT-X-DISCONTINUITY"):
            reset_clock = True
        elif match := re.match(r"#EXTINF:([\d.]+)", line):
            duration = float(match.group(1))
        if line and not line.startswith("#") and pathlib.Path(line).name == line:
            segment = pathlib.Path(directory) / line
            pts = segment_pts(segment)
            if pts is not None:
                if reset_clock:
                    try:
                        segment_time = segment.stat().st_mtime - duration
                        elapsed = (pts - origin_pts) % ((1 << 33) / 90000)
                        if abs(origin_time + elapsed - segment_time) > 60:
                            origin_time = segment_time
                            origin_pts = pts
                    except OSError:
                        pass
                    reset_clock = False
                elapsed = (pts - origin_pts) % ((1 << 33) / 90000)
                date = datetime.fromtimestamp(origin_time + elapsed, UTC)
                result.append("#EXT-X-PROGRAM-DATE-TIME:" + date.isoformat(timespec="milliseconds"))
        result.append(line)
    return "\n".join(result) + "\n"


def aligned(directory: str) -> bool:
    """Do not promote an encoder that is still catching up with the low rendition."""
    try:
        ends = []
        for name in ("low.m3u8", "high.m3u8"):
            content = (pathlib.Path(directory) / name).read_text()
            entries = _PLAYLIST_ENTRIES.findall(content)
            duration, filename = entries[-1]
            pts = segment_pts(pathlib.Path(directory) / filename)
            if pts is None:
                return False
            ends.append(pts + float(duration))
        return abs(ends[0] - ends[1]) <= 3
    except (OSError, ValueError, IndexError):
        return False


def ready_bitrate(
    directory: str,
    name: str,
    *,
    minimum_segments: int = 2,
    max_age: float = 10,
) -> float:
    """Require complete, fresh segments; use the largest recent segment bitrate."""
    path = pathlib.Path(directory) / name
    try:
        if time.time() - path.stat().st_mtime > max_age:
            return 0
        entries = _PLAYLIST_ENTRIES.findall(path.read_text())
        if len(entries) < minimum_segments:
            return 0
        rates = []
        for duration, filename in entries[-3:]:
            if pathlib.Path(filename).name != filename or float(duration) <= 0:
                return 0
            size = (path.parent / filename).stat().st_size
            if size < 1000:
                return 0
            rates.append(size * 8 / float(duration))
        return max(rates)
    except (OSError, ValueError):
        return 0
