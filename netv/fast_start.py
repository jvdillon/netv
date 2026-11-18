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
    deinterlace_fallback: bool,
    *,
    allow_upscale: bool,
    source: MediaInfo | None = None,
    audio_passthrough: bool = False,
    restarting: bool = False,
    rendition: str = "low",
    grid_offset: float = 0.0,
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
        deinterlace_fallback,
        allow_upscale=allow_upscale,
        source=source,
        audio_passthrough=audio_passthrough,
    )
    cmd[cmd.index("-i") : cmd.index("-i")] = ["-f", "mpegts"]
    cmd[cmd.index("-probesize") + 1] = "5000000"
    # The relay starts workers at a keyframe (keyframe_replay). Audio first seen
    # after the window still decodes, unlike remux_command's copy, so a longer one
    # only holds the first segment back (0.8 s measured 0.4 s slower than 0.3 s).
    cmd[cmd.index("-analyzeduration") + 1] = "300000"
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
        # Keyframes on the source clock's quarter-seconds for the short first segments,
        # then its whole seconds, so workers started apart cut at the same timestamps.
        # ``t`` counts from the worker's first frame; ``grid_offset`` is that frame's
        # source time modulo one second.
        "-force_key_frames",
        f"expr:if(isnan(prev_forced_t),1,if(lt(n_forced,4),"
        f"gte(floor((t+{grid_offset:.6f})*4),floor((prev_forced_t+{grid_offset:.6f})*4)+1),"
        f"gte(floor(t+{grid_offset:.6f}),floor(prev_forced_t+{grid_offset:.6f})+1)))",
    ]
    # These encoders mark forced non-IDR I-frames as non-key, so HLS cannot cut there.
    if cmd[cmd.index("-c:v") + 1] in ("h264_nvenc", "libx264"):
        cmd += ["-forced-idr", "1"]
    cmd += _hls_output(directory, restarting=restarting, rendition=rendition)
    limit_live_video_bitrate(cmd, resolution)
    return cmd


def recorder_command(
    directory: str,
    hw: HwAccel,
    resolution: str,
    quality: str,
    seconds: float,
    source: MediaInfo | None,
) -> list[str]:
    """Record the relay as a growing VOD rendition that ends with the program.

    Args:
        directory: Recording directory receiving stream.m3u8.
        hw: Hardware acceleration setting.
        resolution: Maximum output resolution.
        quality: Encoder quality setting.
        seconds: Recording length.
        source: Probed source facts, or None to let ffmpeg probe.

    Returns:
        cmd: FFmpeg arguments reading the relay from stdin.
    """
    cmd = build_hls_ffmpeg_cmd(
        "pipe:0",
        hw,
        directory,
        True,
        [],
        source,
        resolution,
        quality,
        None,
        True,
        allow_upscale=False,
    )
    # Stop reading when the program ends; ffmpeg then closes the playlist.
    cmd[cmd.index("-i") : cmd.index("-i")] = ["-f", "mpegts", "-t", str(int(seconds))]
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
        # Input starts at a keyframe. Copied audio needs its parameters before the
        # header is written, so the window must reach audio muxed ~0.7 s late.
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


def last_media_sequence(content: str) -> int:
    """The media sequence number of a playlist's last segment, or -1 when it lists none.

    Args:
        content: HLS media playlist text.

    Returns:
        msn: ``EXT-X-MEDIA-SEQUENCE`` plus the number of listed segments, minus one.
    """
    first = 0
    segments = 0
    for line in content.splitlines():
        if line.startswith("#EXT-X-MEDIA-SEQUENCE:"):
            first = int(line.removeprefix("#EXT-X-MEDIA-SEQUENCE:"))
        elif line and not line.startswith("#"):
            segments += 1
    return first + segments - 1


# A segment is published only once complete, and a polling client then finds it up to
# a target duration later; with blocking reload it asks for the next one and gets it
# as soon as it is listed.
def with_blocking_reload(content: str) -> str:
    """Advertise blocking playlist reload (LL-HLS ``_HLS_msn``) in a media playlist.

    Args:
        content: HLS media playlist text.

    Returns:
        playlist: ``content`` with an ``EXT-X-SERVER-CONTROL`` tag after its header.
    """
    header = "#EXTM3U\n"
    if not content.startswith(header) or "#EXT-X-SERVER-CONTROL:" in content:
        return content
    return f"{header}#EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES\n{content.removeprefix(header)}"


def blocking_master(content: str) -> str:
    """A master playlist whose rendition is the blocking-reload media playlist.

    Args:
        content: Master playlist listing ``low.m3u8``.

    Returns:
        playlist: The master listing ``low.m3u8?blocking=1`` instead.
    """
    return content.replace("\nlow.m3u8\n", "\nlow.m3u8?blocking=1\n")


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


def playlist_end_pts(directory: str, content: str) -> float | None:
    """Source timestamp where a playlist's last segment ends.

    Args:
        directory: Directory holding the segments.
        content: Media playlist text.

    Returns:
        end: Last segment's first video timestamp plus its duration, or None.
    """
    _, _, segments = _segments(content)
    if not segments:
        return None
    tags, uri = segments[-1]
    start = segment_pts(pathlib.Path(directory) / uri)
    duration = next((float(t[8:].split(",")[0]) for t in tags if t.startswith("#EXTINF:")), 0.0)
    return None if start is None else start + duration


def splice_point(directory: str, low: str, up: str) -> int | None:
    """The upscaled segment that continues the plain rendition's final playlist.

    Both workers cut keyframes on the same source-clock grid, so the upscaler has a
    segment starting exactly where the plain playlist ends; a later one is used if
    the grids ever disagree, so playback never waits on an exact match.

    Args:
        directory: Directory holding both renditions' segments.
        low: The plain rendition's playlist when it stopped being served.
        up: The upscaled rendition's current playlist.

    Returns:
        sequence: The upscaled media sequence number to continue with, or None
            before the upscaler reaches the plain end.
    """
    end = playlist_end_pts(directory, low)
    _, up_sequence, up_segments = _segments(up)
    for index, (_, uri) in enumerate(up_segments):
        start = segment_pts(pathlib.Path(directory) / uri)
        # Encoders round timestamps to their own tick; a millisecond is the same frame.
        if start is not None and (end is None or start >= end - 0.001):
            return up_sequence + index
    return None


def splice_playlists(low: str, up: str, up_from: int | None, *, list_size: int) -> str:
    """Continue the plain rendition with the upscaled one, as a single live playlist.

    Every plain segment clients may have seen keeps its media sequence number; the
    upscaled segments from ``up_from`` follow after a discontinuity, renumbered.

    Args:
        low: The plain rendition's playlist when it stopped being served.
        up: The upscaled rendition's current playlist.
        up_from: ``splice_point``'s result, or None before the upscaler reached it.
        list_size: Segments the live window keeps.

    Returns:
        content: Playlist a client that was playing the plain rendition can continue.
    """
    if up_from is None:
        return low
    header, low_sequence, low_segments = _segments(low)
    _, up_sequence, up_segments = _segments(up)
    combined = [(low_sequence + i, tags, uri) for i, (tags, uri) in enumerate(low_segments)]
    after = low_sequence + len(low_segments)
    for i, (tags, uri) in enumerate(up_segments):
        number = up_sequence + i
        if number >= up_from:
            switch = ["#EXT-X-DISCONTINUITY"] if number == up_from else []
            combined.append((after + number - up_from, switch + tags, uri))
    kept = combined[-list_size:]
    dropped = combined[: len(combined) - len(kept)]
    rolled = sum(tags.count("#EXT-X-DISCONTINUITY") for _, tags, _ in dropped)
    # The upscaler's own window may have passed the switch, taking its tag along.
    rolled += up_sequence > up_from
    lines = [*header, f"#EXT-X-MEDIA-SEQUENCE:{kept[0][0]}"]
    if rolled:
        lines.append(f"#EXT-X-DISCONTINUITY-SEQUENCE:{rolled}")
    for _, tags, uri in kept:
        lines += [*tags, uri]
    return "\n".join(lines) + "\n"


def _hls_output(directory: str, *, restarting: bool, rendition: str = "low") -> list[str]:
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
        f"{directory}/{rendition}_%06d.ts",
        "-hls_flags",
        flags,
        f"{directory}/{rendition}.m3u8",
    ]


# ``playlist_tags`` describe the whole playlist; every other tag belongs to the segment
# after it.
def _segments(
    content: str,
    *,
    playlist_tags: tuple[str, ...] = (
        "#EXTM3U",
        "#EXT-X-VERSION",
        "#EXT-X-TARGETDURATION",
        "#EXT-X-MEDIA-SEQUENCE",
        "#EXT-X-DISCONTINUITY-SEQUENCE",
        "#EXT-X-INDEPENDENT-SEGMENTS",
        "#EXT-X-PLAYLIST-TYPE",
        "#EXT-X-ALLOW-CACHE",
    ),
) -> tuple[list[str], int, list[tuple[list[str], str]]]:
    """Split a media playlist into header lines, first media sequence, and segments."""
    header: list[str] = []
    sequence = 0
    segments: list[tuple[list[str], str]] = []
    tags: list[str] = []
    for line in content.splitlines():
        if line.startswith(playlist_tags):
            if line.startswith("#EXT-X-MEDIA-SEQUENCE:"):
                sequence = int(line.partition(":")[2])
            elif not line.startswith("#EXT-X-DISCONTINUITY-SEQUENCE"):
                header.append(line)
        elif line.startswith("#"):
            tags.append(line)
        elif line:
            segments.append((tags, line))
            tags = []
    return header, sequence, segments
