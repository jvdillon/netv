"""Live fast start: one worker per session, fed by the relay over stdin."""

from datetime import datetime, timezone
from urllib import parse

import math
import pathlib
import re
import time

from netv.cache import HwAccel
from netv.ffmpeg_command import (
    PASSTHROUGH_AUDIO_CODECS,
    MediaInfo,
    build_hls_ffmpeg_cmd,
    get_live_hls_list_size,
    http_reconnect_args,
    limit_live_video_bitrate,
    live_video_bitrates,
)


def live_buffer_durations(
    segment_durations: list[float],
    *,
    playback_buffer_seconds: float = 12.0,
    max_playback_buffer_seconds: float = 60.0,
) -> tuple[float, float]:
    """Return startup reserve and client buffer targets for bursty upstream media.

    Args:
        segment_durations: Observed upstream segment lengths in seconds.
        playback_buffer_seconds: Minimum client playback buffer.
        max_playback_buffer_seconds: Maximum client playback buffer.

    Returns:
        startup_buffer: Reserve covering two upstream segments.
        playback_buffer: Bounded client buffer target in seconds.
    """
    valid = [value for value in segment_durations if math.isfinite(value) and value > 0]
    upstream_duration = max(valid, default=4.0)
    # Live media arrives in real time, so each second of reserve is a second of
    # startup delay. Two upstream segments bridge the gap until the next one.
    startup_buffer = 2 * upstream_duration
    playback_buffer = min(
        max_playback_buffer_seconds,
        max(playback_buffer_seconds, startup_buffer),
    )
    return startup_buffer, playback_buffer


def add_live_start_offset(content: str, seconds: float) -> str:
    """Prefer a safe initial position behind the live edge for native HLS clients.

    Args:
        content: HLS playlist text.
        seconds: Requested distance behind the live edge.

    Returns:
        playlist: Original text or text with a native start-offset tag.
    """
    if not math.isfinite(seconds) or seconds <= 0 or "#EXT-X-START:" in content:
        return content
    line_ending = "\r\n" if content.startswith("#EXTM3U\r\n") else "\n"
    header = f"#EXTM3U{line_ending}"
    if not content.startswith(header):
        return content
    offset = f"{seconds:.3f}".rstrip("0").rstrip(".")
    return content.replace(
        header,
        f"{header}#EXT-X-START:TIME-OFFSET=-{offset},PRECISE=NO{line_ending}",
        1,
    )


def is_hls_url(url: str) -> bool:
    path = parse.urlparse(url).path.lower()
    return path.endswith((".m3u8", ".m3u"))


def source_command(url: str, user_agent: str | None) -> list[str]:
    """Remux an HLS source into the single MPEG-TS byte stream the relay reads.

    Args:
        url: HLS media playlist URL.
        user_agent: User-Agent for upstream requests, or None for ffmpeg's.

    Returns:
        cmd: FFmpeg arguments writing MPEG-TS to stdout.
    """
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", *http_reconnect_args(url, max_delay=5)]
    if user_agent is not None:
        cmd += ["-user_agent", user_agent]
    return [
        *cmd,
        # Providers often serve segments without media extensions.
        "-f",
        "hls",
        "-extension_picky",
        "0",
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
        "mpegts",
        "pipe:1",
    ]


def encoder_command(
    directory: str,
    hw: HwAccel,
    resolution: str,
    quality: str,
    deinterlace: bool,
    *,
    allow_upscale: bool,
    audio: MediaInfo | None = None,
    audio_passthrough: bool = False,
    restarting: bool = False,
    video_codec: str = "",
) -> list[str]:
    cmd = build_hls_ffmpeg_cmd(
        "pipe:0",
        hw,
        directory,
        False,
        [],
        None,
        resolution,
        quality,
        None,
        deinterlace,
        allow_upscale=allow_upscale,
        audio_info=audio,
        audio_passthrough=audio_passthrough,
        video_codec=video_codec,
    )
    cmd[cmd.index("-i") : cmd.index("-i")] = ["-f", "mpegts"]
    cmd[cmd.index("-probesize") + 1] = "5000000"
    # The relay starts workers at a keyframe (keyframe_replay); the window only
    # needs to reach the first audio, muxed up to ~0.7 s behind its video.
    cmd[cmd.index("-analyzeduration") + 1] = "800000"
    # Preserve the source timeline that program dates are derived from.
    cmd[1:1] = ["-copyts"]
    # Every B-frame or lookahead frame holds output back from a starting viewer.
    for option in ("-bf", "-rc-lookahead", "-look_ahead", "-look_ahead_depth"):
        if option in cmd:
            del cmd[cmd.index(option) : cmd.index(option) + 2]
    del cmd[cmd.index("-f", cmd.index("-max_delay")) :]
    cmd += [
        "-bf",
        "0",
        # Four quarter-second keyframes cut the short first segments, then one a second.
        "-force_key_frames",
        "expr:if(isnan(prev_forced_t),1,gte(t,prev_forced_t+if(lt(n_forced,4),0.25,1)))",
    ]
    # These encoders mark forced non-IDR I-frames as non-key, so HLS cannot cut there.
    if cmd[cmd.index("-c:v") + 1] in ("h264_nvenc", "libx264"):
        cmd += ["-forced-idr", "1"]
    cmd += _hls_output(directory, restarting=restarting)
    limit_live_video_bitrate(cmd, resolution)
    return cmd


def remux_command(directory: str, *, restarting: bool = False) -> list[str]:
    """Stream-copy the relay into the DVR-length rendition.

    Takes the encoder's place when the source already plays everywhere: same
    input, output names, segment length, and source timeline, without decoding.

    Args:
        directory: Session directory receiving low.m3u8.
        restarting: Resume an existing rendition after a worker restart.

    Returns:
        cmd: FFmpeg arguments.
    """
    return [
        "ffmpeg",
        "-copyts",
        "-hide_banner",
        "-loglevel",
        "error",
        # Input starts at a keyframe; reach the first audio (see encoder_command).
        "-analyzeduration",
        "800000",
        "-f",
        "mpegts",
        "-i",
        "pipe:0",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-c",
        "copy",
        # The MPEG-TS muxer otherwise delays every timestamp by 2 x max_delay (1.4 s);
        # program dates assume segments keep the source's timestamps.
        "-max_delay",
        "0",
        *_hls_output(directory, restarting=restarting),
    ]


def master_playlist(resolution: str, *, audio_bitrate: int = 0) -> str:
    """Build the master playlist for the session's single local rendition.

    Args:
        resolution: Encoder resolution preset.
        audio_bitrate: Surround bandwidth beyond the usual ten-percent margin.

    Returns:
        playlist: Master playlist advertising low.m3u8.
    """
    _, maximum = live_video_bitrates(resolution)
    # Include room for AAC and MPEG-TS overhead in the advertised bandwidth.
    bandwidth = int(maximum * 1.1) + audio_bitrate
    return f"#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-STREAM-INF:BANDWIDTH={bandwidth}\nlow.m3u8\n"


def closed_caption_master_playlist(resolution: str, *, audio_bitrate: int = 0) -> str:
    """Build a single-rendition master declaring embedded CEA-608/708 captions.

    Args:
        resolution: Video resolution preset.
        audio_bitrate: Additional surround bandwidth in bits per second.

    Returns:
        playlist: Master playlist with an in-stream caption declaration.
    """
    _, maximum = live_video_bitrates(resolution)
    bandwidth = int(maximum * 1.1) + audio_bitrate
    return "\n".join(
        [
            "#EXTM3U",
            "#EXT-X-VERSION:3",
            '#EXT-X-MEDIA:TYPE=CLOSED-CAPTIONS,GROUP-ID="cc",NAME="CC1",'
            'DEFAULT=NO,AUTOSELECT=YES,INSTREAM-ID="CC1"',
            f'#EXT-X-STREAM-INF:BANDWIDTH={bandwidth},CLOSED-CAPTIONS="cc"',
            "low.m3u8",
            "",
        ],
    )


def surround_audio_bitrate(audio: MediaInfo | None, passthrough: bool) -> int:
    """Extra bandwidth to advertise for surround output (Dolby copies at up to 640 kbps).

    Args:
        audio: Probed source audio, or None when unknown.
        passthrough: Whether the client receives Dolby audio as copied.

    Returns:
        bitrate: Bits per second to add to the rendition's advertised bandwidth.
    """
    if audio and passthrough and audio.audio_codec in PASSTHROUGH_AUDIO_CODECS:
        return 640_000
    # An unknown layout (no audio facts, or 0 channels) may encode as 5.1.
    if audio is None or audio.audio_channels not in (1, 2):
        return 384_000
    return 0


def segment_pts(path: pathlib.Path) -> float | None:
    """Read the first video PES presentation timestamp in an MPEG-TS segment.

    Args:
        path: Local MPEG-TS segment.

    Returns:
        timestamp: Presentation time in seconds, or None when unavailable.
    """
    try:
        with path.open("rb") as source:
            return first_video_pts(source.read(188 * 512))
    except OSError:
        return None


# A transport packet starting a table section or PES: sync byte, unit-start set, no error.
_UNIT_START = re.compile(rb"\x47[\x40-\x5f]")


def keyframe_replay(data: bytes) -> bytes | None:
    """MPEG-TS from the last video keyframe, prefixed with the stream's PAT and PMT.

    A worker fed this decodes from its first packet, so its stream analysis is
    nearly instant; nothing before a keyframe is decodable anyway.

    Args:
        data: MPEG-TS packets; may start mid-packet.

    Returns:
        replay: PAT, PMT, and the packets from the keyframe on, or None before
            a keyframe and both tables have arrived.
    """
    sync = next((o for o in range(188) if data[o : o + 377 : 188] == b"\x47\x47\x47"), None)
    if sync is None:
        return None
    pat = pmt = b""
    keyframe = pmt_pid = None
    # Only packets starting a table section or PES matter; the regex finds them in C.
    for match in _UNIT_START.finditer(data, sync):
        offset = match.start()
        if (offset - sync) % 188 or offset + 188 > len(data):
            continue
        packet = data[offset : offset + 188]
        pid = (packet[1] & 0x1F) << 8 | packet[2]
        adaptation = packet[3] & 0x20
        payload = packet[5 + packet[4] :] if adaptation else packet[4:]
        if pid == 0:
            pat = packet
            # Pointer field, 8-byte table header, then the first program's entry.
            entry = payload[1 + payload[0] + 8 : 1 + payload[0] + 12]
            pmt_pid = (entry[2] & 0x1F) << 8 | entry[3] if len(entry) == 4 else pmt_pid
        elif pid == pmt_pid:
            pmt = packet
        elif (
            adaptation
            and packet[4]
            and packet[5] & 0x40  # random_access_indicator.
            and len(payload) > 3
            and payload[:3] == b"\x00\x00\x01"
            and 0xE0 <= payload[3] <= 0xEF  # A video stream id.
        ):
            keyframe = offset
    if keyframe is None or not pat or not pmt:
        return None
    return pat + pmt + data[keyframe:]


def first_video_pts(data: bytes) -> float | None:
    """Read the first video PES presentation timestamp in MPEG-TS bytes.

    Args:
        data: MPEG-TS packets.

    Returns:
        timestamp: Presentation time in seconds, or None when absent.
    """
    for offset in range(0, len(data) - 187, 188):
        packet = data[offset : offset + 188]
        if packet[0] != 0x47 or not packet[1] & 0x40:
            continue
        control = (packet[3] >> 4) & 3
        if control not in (1, 3):
            continue
        start = 4 if control == 1 else 5 + packet[4]
        pes = packet[start:]
        if len(pes) < 14 or pes[:3] != b"\x00\x00\x01" or pes[3] < 0xE0 or pes[3] > 0xEF:
            continue
        if not pes[7] & 0x80:
            continue
        p = pes[9:14]
        pts = (p[0] >> 1 & 7) << 30 | p[1] << 22 | (p[2] >> 1) << 15 | p[3] << 7 | p[4] >> 1
        return pts / 90_000
    return None


def dated_playlist(
    directory: str,
    content: str,
    origin_pts: float,
    origin_time: float,
    dates: dict[str, float],
) -> tuple[str, dict[str, float]]:
    """Map the rendition to the source clock, independent of encoder warm-up.

    A segment keeps the date it was first given: a relay restart moves the
    clock origin, and redating older segments still in the window would move
    them in time.

    Args:
        directory: Directory containing local segment files.
        content: Media playlist text.
        origin_pts: Source presentation timestamp in seconds at origin_time.
        origin_time: Wall-clock time corresponding to origin_pts.
        dates: Segment dates by filename from earlier calls.

    Returns:
        playlist: Media playlist with program dates.
        dates: Dates of the listed segments, for the next call.
    """
    result = []
    listed: dict[str, float] = {}
    for line in content.splitlines():
        if line.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
            continue
        if line and not line.startswith("#") and pathlib.Path(line).name == line:
            date = dates.get(line)
            if date is None and (pts := segment_pts(pathlib.Path(directory) / line)) is not None:
                date = origin_time + (pts - origin_pts) % ((1 << 33) / 90_000)
            if date is not None:
                listed[line] = date
                stamp = datetime.fromtimestamp(date, timezone.utc)
                result.append(
                    "#EXT-X-PROGRAM-DATE-TIME:" + stamp.isoformat(timespec="milliseconds"),
                )
        result.append(line)
    return "\n".join(result) + "\n", listed


def ready_bitrate(
    directory: str,
    name: str,
    *,
    minimum_segments: int = 2,
    max_age: float = 10,
) -> float:
    """Require complete, fresh segments; use the largest recent segment bitrate.

    Args:
        directory: Directory containing playlist and segments.
        name: Media playlist filename.
        minimum_segments: Required number of published segments.
        max_age: Maximum playlist age in seconds.

    Returns:
        bitrate: Largest recent bitrate, or zero when playback is not ready.
    """
    path = pathlib.Path(directory) / name
    try:
        if time.time() - path.stat().st_mtime > max_age:
            return 0
        entries = re.findall(
            r"#EXTINF:([\d.]+),[^\n]*\n(?:#[^\n]*\n)*([^#\n]+)",
            path.read_text(),
        )
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


def _hls_output(directory: str, *, restarting: bool) -> list[str]:
    """HLS muxer options shared by every worker: short segments over the DVR window."""
    duration = 1.0
    flags = "delete_segments+temp_file" + ("+append_list+discont_start" if restarting else "")
    return [
        "-f",
        "hls",
        "-hls_time",
        "1",
        # Playback starts after two segments; short first ones cut the wait.
        "-hls_init_time",
        "0.25",
        "-hls_list_size",
        str(get_live_hls_list_size(duration)),
        "-hls_segment_filename",
        f"{directory}/low_%06d.ts",
        "-hls_flags",
        flags,
        f"{directory}/low.m3u8",
    ]
