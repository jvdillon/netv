"""FFmpeg command building: how a probed source becomes HLS."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import pairwise
from typing import Final

import logging
import pathlib
import subprocess
import tempfile

from netv import cache
from netv.cache import HwAccel, ServerSettings
from netv.ffmpeg import probe
from netv.ffmpeg.letterbox import CropBox, detection_args
from netv.ffmpeg.probe import MediaInfo, SubtitleStream


log = logging.getLogger(__name__)

# Segment names are a persistence-format contract shared by command and session readers.
SEG_PREFIX: Final = "seg"

# Dolby audio that Apple players decode natively; other clients need AAC.
PASSTHROUGH_AUDIO_CODECS = frozenset({"ac3", "eac3"})

_gpu_nvdec_codecs: set[str] | None = None  # None = not probed yet.

_has_libplacebo: bool | None = None  # None = not probed yet.

# The loader main injects; init() replaces it (the gateway layers upscale checks on top).
_load_settings: Callable[[], ServerSettings] = cache.load_server_settings

# Directory containing TensorRT engines; None until init() enables AI Upscale.
_sr_engine_dir: str | None = None


def init(
    load_settings: Callable[[], ServerSettings],
    sr_engine_dir: str = "",
) -> None:
    """Initialize module with settings loader and optional AI Upscale config.

    Args:
        load_settings: Callable returning current server settings.
        sr_engine_dir: TensorRT engine directory; empty disables AI Upscale.
    """
    global _load_settings, _sr_engine_dir
    cache.initialize()
    _load_settings = load_settings
    _sr_engine_dir = sr_engine_dir or None
    probe.load_caches()
    if _sr_engine_dir is not None:
        log.info("AI Upscale enabled: engine_dir=%s", _sr_engine_dir)


def get_settings() -> ServerSettings:
    """Get current settings."""
    return _load_settings()


def get_hls_segment_duration() -> float:
    """Get HLS segment duration in seconds."""
    # Short segments start and seek faster.
    return 3.0


def get_transcode_dir() -> pathlib.Path:
    """Get the transcode output directory, falling back to system temp.

    Returns:
        directory: Configured accessible directory, or the system temp directory.
    """
    custom_dir = _load_settings().get("transcode_dir", "")
    if custom_dir:
        path = pathlib.Path(custom_dir)
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except (PermissionError, OSError) as e:
            log.warning("Transcode dir %s inaccessible (%s), using temp dir", custom_dir, e)
    return pathlib.Path(tempfile.gettempdir())


def http_reconnect_args(url: str, *, max_delay: int = 30) -> list[str]:
    """Build reconnect options for HTTP inputs only.

    Args:
        url: FFmpeg input URL or path.
        max_delay: Maximum reconnect delay in seconds.

    Returns:
        args: Reconnect options, empty for local or non-HTTP inputs.
    """
    if not url.startswith(("http://", "https://")):
        return []
    return [
        "-reconnect",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_on_network_error",
        "1",
        "-reconnect_on_http_error",
        "4xx,5xx",
        "-reconnect_delay_max",
        str(max_delay),
    ]


def live_video_bitrates(resolution: str) -> tuple[int, int]:
    """Return target and maximum video bitrate shared by live renditions.

    Args:
        resolution: Rendition resolution preset; unknown presets use 720p rates.

    Returns:
        target: Average video bitrate in bits per second.
        maximum: Peak video bitrate and buffer size in bits per second.
    """
    return {
        "480p": (2_000_000, 3_000_000),
        "720p": (4_000_000, 6_000_000),
        "1080p": (6_000_000, 8_000_000),
        "4k": (16_000_000, 20_000_000),
    }.get(resolution, (4_000_000, 6_000_000))


def limit_live_video_bitrate(cmd: list[str], resolution: str) -> None:
    """Replace quality-only rate control with bounded live bitrates, in place.

    Args:
        cmd: FFmpeg command whose final element is the output path.
        resolution: Rendition resolution preset selecting the bitrate bounds.
    """
    target, maximum = live_video_bitrates(resolution)
    encoder = cmd[cmd.index("-c:v") + 1]
    for flag in (
        "-rc",
        "-rc_mode",
        "-qp",
        "-qp_i",
        "-qp_p",
        "-global_quality",
        "-crf",
        "-b:v",
        "-maxrate",
        "-bufsize",
    ):
        if flag in cmd:
            index = cmd.index(flag)
            del cmd[index : index + 2]
    rate_mode = {
        "h264_nvenc": ["-rc", "vbr"],
        "h264_amf": ["-rc", "vbr_peak"],
        "h264_vaapi": ["-rc_mode", "VBR"],
    }.get(encoder, [])
    cmd[-1:-1] = rate_mode + [
        "-b:v",
        str(target),
        "-maxrate",
        str(maximum),
        "-bufsize",
        str(maximum),
    ]


def get_live_hls_list_size(
    segment_duration: float = 3.0,
    *,
    live_buffer_secs: float = 30.0,
) -> int:
    """Get hls_list_size for live streams based on DVR setting.

    Args:
        segment_duration: HLS segment length in seconds.
        live_buffer_secs: Retained live window when DVR is disabled.

    Returns:
        list_size: Number of segments retained in the live playlist.
    """
    dvr_mins = _load_settings()["live_dvr_mins"]
    if dvr_mins <= 0:
        return int(live_buffer_secs / segment_duration)
    # DVR enabled: calculate segments from minutes.
    return int(dvr_mins * 60 / segment_duration)


def uses_audio_passthrough(media_info: MediaInfo | None, passthrough: bool) -> bool:
    """Will the output carry Dolby audio that only passthrough-capable clients can play?"""
    return bool(passthrough and media_info and media_info.audio_codec in PASSTHROUGH_AUDIO_CODECS)


def can_remux_live(
    media_info: MediaInfo | None,
    max_resolution: str = "1080p",
    *,
    audio_passthrough: bool = False,
) -> bool:
    """Can a live source be retained via a stream-copy HLS remux?"""
    return _can_copy_video(media_info, max_resolution) and _can_copy_audio(
        media_info,
        passthrough=audio_passthrough,
    )


def encoder_crop(
    media_info: MediaInfo | None,
    max_resolution: str,
    *,
    allow_upscale: bool = True,
) -> CropBox | None:
    """The crop a transcode of ``media_info`` applies to cut its black bars.

    Args:
        media_info: Probed source facts with any detected picture area.
        max_resolution: Maximum output resolution preset.
        allow_upscale: Whether AI Upscale may apply.

    Returns:
        crop: The detected picture area, or None when there are no bars, no frame
            size to place them in, or the video is copied rather than re-encoded.
    """
    if _can_copy_video(media_info, max_resolution, allow_upscale=allow_upscale):
        return None
    return _cuttable_crop(media_info)


def build_hls_ffmpeg_cmd(
    input_url: str,
    hw: HwAccel,
    output_dir: str,
    is_vod: bool = False,
    subtitles: list[SubtitleStream] | None = None,
    media_info: MediaInfo | None = None,
    max_resolution: str = "1080p",
    quality: str = "high",
    user_agent: str | None = None,
    deinterlace_fallback: bool | None = None,
    allow_upscale: bool = True,
    source: MediaInfo | None = None,
    audio_passthrough: bool = False,
    *,
    input_args: Sequence[str],
    sr_segment_duration: float = 2.0,
    vod_read_burst_sec: int = 120,
) -> list[str]:
    """Build ffmpeg command for HLS transcoding.

    Args:
        input_url: Source URL or local path.
        hw: Encoder and decode-fallback selection.
        output_dir: Directory receiving playlists and segments.
        is_vod: Whether to keep a complete event playlist.
        subtitles: Text subtitle streams to extract.
        media_info: Probed source facts; None adds probe options to the command.
        max_resolution: Maximum output resolution preset.
        quality: Quality preset for constant-quality encoders.
        user_agent: HTTP User-Agent for the input.
        deinterlace_fallback: Deinterlace when probe data is unavailable.
        allow_upscale: Whether AI Upscale may apply.
        source: Facts probed from a sample of an input the command still probes and
            re-encodes (a live relay); used in place of ``media_info`` for every
            decision but stream copy.
        audio_passthrough: Copy Dolby audio for clients that decode it natively.
        input_args: Options that keep the input's connections behind the egress
            proxy (``proxy.ffmpeg_input_args``); none for a pipe.
        sr_segment_duration: HLS segment length when AI Upscale applies.
        vod_read_burst_sec: Seconds of a network VOD input read at full speed
            before reading holds to real time.

    Returns:
        cmd: FFmpeg arguments.
    """
    # Every decision below reads this one record, so no caller can pass some probed
    # facts and drop others. Only stream copy needs media_info: a ``source`` is re-encoded.
    probed = media_info or source
    copy_video = _can_copy_video(media_info, max_resolution, allow_upscale=allow_upscale)
    copy_audio = _can_copy_audio(probed, passthrough=audio_passthrough)
    # Only a probed file's bars are cut: a live sample shows one moment of a channel.
    # A copied stream ignores it (encoder_crop).
    crop = _cuttable_crop(media_info)
    crop_pixels = (
        crop.pixels(media_info.width, media_info.height)
        if crop is not None and media_info is not None
        else None
    )

    # Full hardware pipeline if GPU supports the codec
    # Parse hw to get encoder type.
    enc_type, _ = _parse_hw(hw)
    codec = probed.video_codec if probed else ""
    use_hw_pipeline = bool(
        not copy_video
        and codec
        and (
            (enc_type == "nvenc" and codec in _get_gpu_nvdec_codecs())
            # VAAPI/QSV lack a portable runtime probe (vainfo output varies by
            # driver), so use codecs nearly universal on GPUs from the last decade.
            or (
                enc_type == "vaapi"
                and codec in {"h264", "hevc", "mpeg2video", "vp8", "vp9", "vc1", "av1"}
            )
            or (enc_type == "qsv" and codec in {"h264", "hevc", "mpeg2video", "vp9", "vc1", "av1"})
            # AMF never has hw decode pipeline - always False.
        ),
    )

    # Deinterlace: use probe result if available, else use fallback setting
    # (fallback defaults to True for live, False for VOD when not explicitly set)
    fallback = deinterlace_fallback if deinterlace_fallback is not None else (not is_vod)
    # If probe failed (height=0), don't trust interlaced flag - use fallback.
    # A live sample shows only the moment it was taken: a channel airing a progressive
    # segment switches to 60i shows later, so a raster that can carry fields keeps the
    # deinterlacer, which passes frames marked progressive through untouched.
    deinterlace = (
        (probed.interlaced or (probed is source and _can_carry_fields(probed)))
        if probed and probed.height > 0
        else fallback
    )

    # Build component arg lists.
    video_pre, video_post = _build_video_args(
        copy_video=copy_video,
        hw=hw,
        deinterlace=deinterlace,
        use_hw_pipeline=use_hw_pipeline,
        max_resolution=max_resolution,
        quality=quality,
        is_hdr=probed.is_hdr if probed else False,
        source_height=(
            crop_pixels[1] if crop_pixels is not None else (probed.height if probed else 0)
        ),
        allow_upscale=allow_upscale,
        crop=crop_pixels,
    )
    audio_args = _build_audio_args(copy_audio=copy_audio, audio=probed)
    sr_applied = any("dnn_processing=" in arg for arg in video_post)
    segment_duration = sr_segment_duration if sr_applied else get_hls_segment_duration()

    # Base args.
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-noautorotate",
    ]

    # Hwaccel args (before -i)
    cmd.extend(video_pre)

    # Probe args (only when no media_info, since we already probed)
    if media_info is None:
        probe_size = "50000" if is_vod else "5000000"
        analyze_dur = "500000" if is_vod else "5000000"
        cmd.extend(["-probesize", probe_size, "-analyzeduration", analyze_dur])

    # Input args.
    cmd.extend(
        [
            "-fflags",
            "+discardcorrupt+genpts",
            "-err_detect",
            "ignore_err",
        ],
    )
    cmd.extend(input_args)
    cmd.extend(http_reconnect_args(input_url))
    if is_vod and input_url.startswith(("http://", "https://")):
        # Copying video outruns playback many times over and fills the transcode dir
        # (often a RAM tmpfs) with a whole movie; stay a burst ahead of real time instead.
        cmd.extend(
            [
                "-readrate",
                "1",
                "-readrate_initial_burst",
                str(vod_read_burst_sec),
                "-readrate_catchup",
                "2",
            ],
        )
    if user_agent is not None:
        cmd.extend(["-user_agent", user_agent])
    # Use HLS demuxer options if probe detected HLS format.
    if media_info and media_info.is_hls:
        cmd.extend(["-f", "hls", "-extension_picky", "0"])
    cmd.extend(["-i", input_url])

    # Keep the growing VTT files for web playback and package the same tracks
    # as HLS subtitle renditions for native players.
    for i, sub in enumerate(subtitles or []):
        cmd.extend(
            [
                "-map",
                f"0:{sub.index}",
                "-c:s",
                "webvtt",
                "-flush_packets",
                "1",
                f"{output_dir}/sub{i}.vtt",
                "-map",
                f"0:{sub.index}",
                "-c:s",
                "webvtt",
                "-f",
                "segment",
                "-segment_time",
                str(int(segment_duration)),
                "-segment_list",
                f"{output_dir}/sub{i}.m3u8",
                "-segment_list_type",
                "m3u8",
                "-segment_list_size",
                "0" if is_vod else str(get_live_hls_list_size(segment_duration)),
                "-segment_format",
                "webvtt",
            ],
        )
        if not is_vod:
            cmd.extend(["-segment_list_flags", "+live"])
        cmd.append(f"{output_dir}/sub{i}_%06d.vtt")

    # A copied stream is never decoded, so it has no frames to look for bars in.
    if not copy_video:
        hw_frames = next(
            (value for flag, value in pairwise(video_pre) if flag == "-hwaccel_output_format"),
            "",
        )
        cmd.extend(
            detection_args(output_dir, hw_frames, is_10bit=bool(probed and probed.is_10bit)),
        )

    # Stream mapping + video + audio.
    audio_index = media_info.audio_index if media_info else None
    cmd.extend(["-map", "0:v:0", "-map", "0:a:0" if audio_index is None else f"0:{audio_index}"])
    cmd.extend(video_post)
    cmd.extend(audio_args)

    # HLS output args.
    cmd.extend(
        [
            "-max_delay",
            "5000000",
            "-f",
            "hls",
            "-hls_time",
            str(int(segment_duration)),
            "-hls_list_size",
            "0" if is_vod else str(get_live_hls_list_size(segment_duration)),
            "-hls_segment_filename",
            f"{output_dir}/{SEG_PREFIX}%03d.ts",
        ],
    )
    if is_vod:
        cmd.extend(
            [
                "-hls_init_time",
                "2",
                "-hls_flags",
                "independent_segments",
                "-hls_playlist_type",
                "event",
            ],
        )
    else:
        # Startup waits for two segments; short first ones (cut at the next
        # keyframe) let live playback begin without waiting out two full ones.
        cmd.extend(["-hls_init_time", "1", "-hls_flags", "delete_segments"])

    cmd.append(f"{output_dir}/stream.m3u8")
    if sr_applied and not is_vod:
        limit_live_video_bitrate(cmd, max_resolution)
    return cmd


def _parse_hw(hw: str) -> tuple[str, str]:
    """Parse hw into (encoder, fallback). e.g. 'nvenc+vaapi' -> ('nvenc', 'vaapi')"""
    if "+" in hw:
        encoder, fallback = hw.split("+", 1)
        return encoder, fallback
    return hw, "software"  # Standalone options fallback to software.


# Returns (engine_path, input_height, input_width, scale_factor) or None if not found.
def _find_sr_engine(
    source_height: int,
    *,
    model_name: str = "2x-nomosuni-compact",
) -> tuple[str, int, int, int] | None:
    """Find the best matching engine file for the given resolution."""
    if _sr_engine_dir is None:
        return None
    engine_dir = pathlib.Path(_sr_engine_dir)
    if not engine_dir.exists():
        return None

    # Engine naming: {model_name}_{height}p_fp16.engine.
    engines: list[tuple[int, pathlib.Path]] = []
    for engine in engine_dir.glob(f"{model_name}_*p_fp16.engine"):
        # Extract height from filename.
        name = engine.stem
        parts = name.rsplit("_", 2)
        if len(parts) >= 3:
            height_str = parts[1].rstrip("p")
            if height_str.isdigit():
                engines.append((int(height_str), engine))

    if not engines:
        return None

    # Sort by height ascending.
    engines.sort(key=lambda x: x[0])

    # Select appropriate engine based on source height.
    if source_height <= 0:
        # Probe failed - use highest resolution engine.
        engine_height, engine_path = engines[-1]
        log.warning("SR: probe failed, using %dp engine", engine_height)
    else:
        # Find engine closest to but >= source height, or use largest if source is bigger.
        engine_height, engine_path = engines[-1]  # Default to largest.
        for h, p in engines:
            if h >= source_height:
                engine_height, engine_path = h, p
                break

    # Calculate width assuming 16:9 aspect ratio, rounded to multiple of 8.
    engine_width = ((engine_height * 16 // 9) + 7) // 8 * 8

    return str(engine_path), engine_height, engine_width, 2


# SR is controlled by sr_model setting - if a model is selected, SR is applied when
# source height < target height.
def _build_sr_filter(
    source_height: int,
    target_height: int,
    *,
    model_name: str = "2x-nomosuni-compact",
    aspect: float = 0.0,
) -> str:
    """Build AI Upscale filter string if needed; aspect > 0 crops away the engine's padding."""
    if _sr_engine_dir is None:
        return ""

    if not _load_settings().get("sr_model", ""):
        return ""  # SR disabled (Off selected)

    engine_info = _find_sr_engine(source_height, model_name=model_name)
    if not engine_info:
        log.warning("SR: no engine found for model=%s, source=%dp", model_name, source_height)
        return ""

    engine_path, engine_height, engine_width, scale = engine_info

    # Apply SR when source resolution is below target (upscaling scenario)
    if target_height and source_height >= target_height:
        log.info(
            "SR: skipping %s - source %dp >= target %dp",
            model_name,
            source_height,
            target_height,
        )
        return ""

    log.info(
        "SR: applying %s (%dx) to %dp -> %dp",
        model_name,
        scale,
        source_height,
        target_height or (source_height * scale),
    )

    # Build SR filter chain:
    # 1. Scale to engine's expected input size (preserving aspect with padding if needed)
    # 2. Convert to RGB (model expects 3-channel RGB input)
    # 3. hwupload to GPU (critical for performance - keeps data on GPU)
    # 4. Apply SR via TensorRT dnn_processing (outputs Nx resolution on GPU)
    # 5. Scale down on GPU to target resolution.
    sr_filter = (
        f"scale={engine_width}:{engine_height}:force_original_aspect_ratio=decrease,"
        f"pad={engine_width}:{engine_height}:(ow-iw)/2:(oh-ih)/2,"
        f"format=rgb24,"
        f"hwupload,"
        f"dnn_processing=dnn_backend=tensorrt:model={engine_path}"
    )
    if aspect:
        # The engine's 16:9 input pads a cropped picture back out; cut that padding on
        # the GPU. scale_cuda sizes from the padded frame, so its size is explicit.
        content_w = min(engine_width, round(engine_height * aspect))
        content_h = min(engine_height, round(engine_width / aspect))
        out_h = target_height or content_h * scale
        x = (engine_width - content_w) // 2 * scale
        y = (engine_height - content_h) // 2 * scale
        return (
            f"{sr_filter},crop={content_w * scale}:{content_h * scale}:{x}:{y},"
            f"scale_cuda=w={round(out_h * aspect / 2) * 2}:h={out_h}"
        )
    if target_height:
        # After dnn_processing, data is on GPU - use scale_cuda with explicit params.
        sr_filter += f",scale_cuda=w=-2:h={target_height}"
    return sr_filter


def _get_gpu_nvdec_codecs() -> set[str]:
    """Get supported NVDEC codecs, probing GPU on first call."""
    global _gpu_nvdec_codecs
    if _gpu_nvdec_codecs is not None:
        return _gpu_nvdec_codecs
    _gpu_nvdec_codecs = set()
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            log.info("No NVIDIA GPU detected")
            return _gpu_nvdec_codecs
        # Parse "NVIDIA GeForce GTX TITAN X, 5.2"
        line = result.stdout.strip().split("\n")[0]
        parts = line.rsplit(",", 1)
        if len(parts) != 2:
            return _gpu_nvdec_codecs
        gpu_name = parts[0].strip()
        compute_cap = float(parts[1].strip())
        # Minimum compute capability per NVDEC codec:
        # https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new
        # HEVC 10-bit requires Pascal; the Maxwell GM206 edge case is ignored. Every
        # NVDEC (Kepler on) decodes MPEG-2, the codec of US broadcast TV.
        nvdec_min_compute = {"mpeg2video": 3.0, "h264": 5.0, "hevc": 6.0, "av1": 8.0}
        _gpu_nvdec_codecs = {
            codec for codec, min_cap in nvdec_min_compute.items() if compute_cap >= min_cap
        }
        log.info(
            "GPU: %s (compute %.1f) NVDEC: %s",
            gpu_name,
            compute_cap,
            ", ".join(sorted(_gpu_nvdec_codecs)) or "none",
        )
    except Exception as e:
        log.debug("GPU probe failed: %s", e)
    return _gpu_nvdec_codecs


def _has_libplacebo_filter() -> bool:
    """Check if FFmpeg has libplacebo filter available (for GPU HDR tone mapping)."""
    global _has_libplacebo
    if _has_libplacebo is not None:
        return _has_libplacebo
    _has_libplacebo = False
    try:
        result = subprocess.run(
            ["ffmpeg", "-filters"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        _has_libplacebo = "libplacebo" in result.stdout
        log.info("libplacebo filter available: %s", _has_libplacebo)
    except Exception as e:
        log.debug("libplacebo probe failed: %s", e)
    return _has_libplacebo


# Returns (pre_input_args, post_input_args).
def _build_video_args(
    *,
    copy_video: bool,
    hw: str,
    deinterlace: bool,
    use_hw_pipeline: bool,
    max_resolution: str,
    quality: str,
    is_hdr: bool = False,
    source_height: int = 0,
    allow_upscale: bool = True,
    sr_filter_threads: int = 4,
    crop: tuple[int, int, int, int] | None = None,
) -> tuple[list[str], list[str]]:
    """Build video args, cutting ``crop`` (width, height, x, y) from each frame."""
    if copy_video:
        return [], ["-c:v", "copy"]

    # Parse hw into encoder and fallback.
    enc_type, fallback = _parse_hw(hw)

    max_h = _max_resolution_height(max_resolution)

    # AI Upscale runs on NVENC only; it needs CPU frames, so it turns off the hw pipeline.
    sr_filter = (
        _build_sr_filter(source_height, max_h or 0, aspect=crop[0] / crop[1] if crop else 0.0)
        if allow_upscale and enc_type == "nvenc"
        else ""
    )
    use_hw_pipeline = use_hw_pipeline and not sr_filter
    # Crops are measured on CPU and CUDA frames only; VAAPI and QSV frames decode in
    # software instead, since an ignored crop there would squash the picture.
    if crop is not None:
        use_hw_pipeline = use_hw_pipeline and enc_type == "nvenc"
    cpu_crop = "" if crop is None or use_hw_pipeline else "crop={}:{}:{}:{},".format(*crop)

    # Fall back gracefully if VAAPI is needed but no device was detected.
    vaapi_device = cache.VAAPI_DEVICE or ""
    vaapi_decode = fallback == "vaapi" and bool(vaapi_device) and crop is None
    if not vaapi_device:
        if enc_type == "vaapi":
            log.warning(
                "VAAPI unavailable (no Intel/AMD GPU), falling back to software encoding",  # pragma: no mutate[diagnostic-text]
            )
            enc_type = "software"
        elif fallback == "vaapi":
            log.warning(
                "VAAPI fallback unavailable (no Intel/AMD GPU), using software decode",  # pragma: no mutate[diagnostic-text]
            )

    # Height expr for scale filter (scale down only, -2 keeps width divisible by 2)
    h = "" if max_h is None else f"min(ih\\,{max_h})"
    # Lower QP/CRF means higher quality.
    qp = {"high": 20, "low": 35}.get(quality, 28)
    pre: list[str]

    if enc_type == "nvenc":
        if use_hw_pipeline:
            # CUDA decode path.
            pre = [
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-extra_hw_frames",
                "3",
            ]
            scale = f"scale_cuda=-2:{h}:format=nv12" if h else "scale_cuda=format=nv12"
            if crop is not None:
                # A crop on CUDA frames only marks them; scale_cuda cuts it, but sizes
                # from the uncropped frame, so -2 would squash the picture.
                crop_w, crop_h, x, y = crop
                out_h = crop_h if max_h is None else min(crop_h, max_h)
                out_w = round(crop_w * out_h / crop_h / 2) * 2
                scale = f"crop={crop_w}:{crop_h}:{x}:{y},scale_cuda={out_w}:{out_h}:format=nv12"
            deint = (
                "yadif_cuda=0:-1:1," if deinterlace else ""
            )  # One frame per frame; frames marked progressive pass.
            # HDR tone mapping: prefer libplacebo (Vulkan GPU), fall back to CPU zscale+tonemap.
            if is_hdr:
                if _has_libplacebo_filter():
                    tonemap = "hwdownload,format=p010le,libplacebo=tonemapping=hable:colorspace=bt709:color_primaries=bt709:color_trc=bt709,format=nv12,hwupload_cuda,"
                else:
                    tonemap = "hwdownload,format=p010le,zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=nv12,hwupload_cuda,"
            else:
                tonemap = ""
            vf = f"{deint}{tonemap}{scale}"
        elif vaapi_decode:
            # VAAPI decode + VAAPI filters + hwdownload + hwupload_cuda for NVENC.
            pre = [
                "-hwaccel",
                "vaapi",
                "-hwaccel_output_format",
                "vaapi",
                "-hwaccel_device",
                vaapi_device,
            ]
            scale = f"scale_vaapi=w=-2:h={h}:format=nv12" if h else "scale_vaapi=format=nv12"
            tonemap = "tonemap_vaapi=format=nv12:t=bt709:m=bt709:p=bt709," if is_hdr else ""
            deint = "deinterlace_vaapi=auto=1," if deinterlace else ""
            vf = f"{deint}{tonemap}{scale},hwdownload,format=nv12,hwupload_cuda"
        else:
            # Software decode, upload to GPU for scaling/encoding.
            pre = []
            scale = f"scale_cuda=-2:{h}:format=nv12" if h else "scale_cuda=format=nv12"
            # HDR tone mapping: prefer libplacebo (Vulkan GPU), fall back to CPU zscale+tonemap
            # Deinterlace before tonemap (CPU yadif) for consistency with hw decode path.
            if sr_filter:
                # SR path: CPU decode -> deinterlace -> SR (GPU) -> encode
                # SR filter ends with scale_cuda, outputs CUDA frames ready for nvenc
                # Need init_hw_device for TensorRT dnn_processing to use GPU.
                pre = [
                    "-init_hw_device",
                    "cuda=cu",
                    "-filter_hw_device",
                    "cu",
                    "-filter_threads",
                    str(sr_filter_threads),
                ]
                deint = "yadif=0:-1:1," if deinterlace else ""
                vf = f"{deint}{sr_filter}"
            elif is_hdr:
                deint = "yadif=0:-1:1," if deinterlace else ""  # CPU deinterlace before tonemap.
                if _has_libplacebo_filter():
                    tonemap = "libplacebo=tonemapping=hable:colorspace=bt709:color_primaries=bt709:color_trc=bt709,format=nv12,hwupload_cuda,"
                else:
                    tonemap = "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=nv12,hwupload_cuda,"
                vf = f"{deint}{tonemap}{scale}"
            else:
                deint = "yadif_cuda=0:-1:1," if deinterlace else ""  # GPU deinterlace after upload.
                tonemap = "format=nv12,hwupload_cuda,"
                vf = f"{tonemap}{deint}{scale}"
        preset = "p5" if sr_filter else ("p4" if deinterlace else "p2")
        encoder = "h264_nvenc"
        enc_opts = [
            "-preset",
            preset,
            "-rc",
            "constqp",
            "-qp",
            str(qp),
            "-bf",
            "0" if sr_filter else "3",
            "-spatial-aq",
            "1",
            "-temporal-aq",
            "1",
        ]
        if not sr_filter:
            enc_opts[6:6] = ["-rc-lookahead", "32"]

    elif enc_type == "amf":
        # AMF has no hardware decode - always uses fallback for decode/filter.
        if vaapi_decode:
            # VAAPI decode + VAAPI filters + hwdownload for AMF encode.
            pre = [
                "-hwaccel",
                "vaapi",
                "-hwaccel_output_format",
                "vaapi",
                "-hwaccel_device",
                vaapi_device,
            ]
            scale = f"scale_vaapi=w=-2:h={h}:format=nv12" if h else "scale_vaapi=format=nv12"
            tonemap = "tonemap_vaapi=format=nv12:t=bt709:m=bt709:p=bt709," if is_hdr else ""
            deint = "deinterlace_vaapi=auto=1," if deinterlace else ""
            vf = f"{deint}{tonemap}{scale},hwdownload,format=nv12"
        else:
            # Software decode + software filters.
            pre = []
            if is_hdr:
                tonemap = "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=nv12,"
            else:
                tonemap = ""
            deint = "yadif=0:-1:1," if deinterlace else ""
            scale = f"scale=-2:{h}," if h else ""
            vf = f"{deint}{tonemap}{scale}format=nv12"
        encoder = "h264_amf"
        enc_opts = [
            "-rc",
            "cqp",
            "-qp_i",
            str(qp),
            "-qp_p",
            str(qp),
            "-quality",
            "balanced",
        ]

    elif enc_type == "vaapi":
        if use_hw_pipeline:
            pre = [
                "-hwaccel",
                "vaapi",
                "-hwaccel_output_format",
                "vaapi",
                "-hwaccel_device",
                vaapi_device,
                "-extra_hw_frames",
                "3",
            ]
            scale = f"scale_vaapi=w=-2:h={h}:format=nv12" if h else "scale_vaapi=format=nv12"
            # HDR tone mapping on VAAPI.
            tonemap = "tonemap_vaapi=format=nv12:t=bt709:m=bt709:p=bt709," if is_hdr else ""
            vf = (
                f"deinterlace_vaapi=auto=1,{tonemap}{scale}" if deinterlace else f"{tonemap}{scale}"
            )
        else:
            # Software decode, upload to GPU for scaling/encoding.
            pre = ["-vaapi_device", vaapi_device]
            scale = f"scale_vaapi=w=-2:h={h}:format=nv12" if h else "scale_vaapi=format=nv12"
            tonemap = "tonemap_vaapi=format=nv12:t=bt709:m=bt709:p=bt709," if is_hdr else ""
            deint = "deinterlace_vaapi=auto=1," if deinterlace else ""
            vf = f"format=nv12,hwupload,{deint}{tonemap}{scale}"
        encoder = "h264_vaapi"
        # Use baseline profile for older GPUs (e.g., AMD GCN 1.0) that don't support High profile.
        if cache.AVAILABLE_ENCODERS.get("vaapi_baseline_only") is True:
            # Baseline doesn't support B-frames.
            enc_opts = ["-rc_mode", "CQP", "-qp", str(qp), "-profile:v", "constrained_baseline"]
        else:
            enc_opts = ["-rc_mode", "CQP", "-qp", str(qp), "-bf", "3"]

    elif enc_type == "qsv":
        if use_hw_pipeline:
            pre = ["-hwaccel", "qsv", "-hwaccel_output_format", "qsv"]
            scale = f"scale_qsv=w=-2:h={h}:format=nv12" if h else "scale_qsv=format=nv12"
            # Combine deinterlace and tonemap into single vpp_qsv call when possible.
            if deinterlace and is_hdr:
                vf = f"vpp_qsv=deinterlace=2:tonemap=1:format=nv12,{scale}"
            elif deinterlace:
                vf = f"vpp_qsv=deinterlace=2,{scale}"
            elif is_hdr:
                vf = f"vpp_qsv=tonemap=1:format=nv12,{scale}"
            else:
                vf = scale
        else:
            # Software decode, upload to GPU for scaling/encoding.
            pre = ["-init_hw_device", "qsv=hw", "-filter_hw_device", "hw"]
            scale = f"scale_qsv=w=-2:h={h}:format=nv12" if h else "scale_qsv=format=nv12"
            # Combine deinterlace and tonemap into single vpp_qsv call when possible.
            if deinterlace and is_hdr:
                vf = f"format=nv12,hwupload=extra_hw_frames=64,vpp_qsv=deinterlace=2:tonemap=1:format=nv12,{scale}"
            elif deinterlace:
                vf = f"format=nv12,hwupload=extra_hw_frames=64,vpp_qsv=deinterlace=2,{scale}"
            elif is_hdr:
                vf = (
                    f"format=nv12,hwupload=extra_hw_frames=64,vpp_qsv=tonemap=1:format=nv12,{scale}"
                )
            else:
                vf = f"format=nv12,hwupload=extra_hw_frames=64,{scale}"
        encoder = "h264_qsv"
        enc_opts = [
            "-global_quality",
            str(qp),
            "-bf",
            "3",
            "-look_ahead",
            "1",
            "-look_ahead_depth",
            "40",
        ]

    elif enc_type == "software":
        pre = []
        # HDR tone mapping on CPU.
        if is_hdr:
            tonemap = "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p,"
        else:
            tonemap = ""
        deint = (
            "yadif=0:-1:1," if deinterlace else ""
        )  # One frame per frame; frames marked progressive pass.
        if h:
            vf = f"{deint}{tonemap}scale=-2:{h},format=yuv420p"
        else:
            vf = f"{deint}{tonemap}format=yuv420p"
        crf = {"high": 20, "low": 32}.get(quality, 26)
        encoder = "libx264"
        enc_opts = ["-preset", "veryfast", "-crf", str(crf), "-bf", "3"]

    else:
        raise ValueError(f"Unrecognized hardware encoder: '{enc_type}'.")

    # Film with pulldown decodes slower than the stream's rate; left to the muxer,
    # the missing frames are duplicated with their CEA-608 bytes, garbling captions.
    # The fps filter fills them and re-paces the caption data.
    post = ["-vf", f"fps=source_fps,{cpu_crop}{vf}", "-c:v", encoder, *enc_opts, "-g", "60"]
    return pre, post


def _build_audio_args(*, copy_audio: bool, audio: MediaInfo | None) -> list[str]:
    """Build audio args."""
    if copy_audio:
        return ["-c:a", "copy"]
    audio_sample_rate = audio.audio_sample_rate if audio else None
    audio_channels = audio.audio_channels if audio else 0
    rate = str(audio_sample_rate) if audio_sample_rate in (44_100, 48_000) else "48000"
    if audio_channels > 2:
        # Every client decodes 5.1 AAC; wider layouts fold down to 5.1.
        return [
            "-c:a",
            "aac",
            "-af",
            "aformat=channel_layouts=5.1",
            "-ar",
            rate,
            "-b:a",
            "384k",
            "-profile:a",
            "aac_low",
        ]
    if audio_channels == 0:
        # Unknown layout (not probed): keep surround as 5.1, everything else as stereo.
        return [
            "-c:a",
            "aac",
            "-af",
            "aformat=channel_layouts=stereo|5.1",
            "-ar",
            rate,
            "-b:a",
            "192k",
            "-profile:a",
            "aac_low",
        ]
    return ["-c:a", "aac", "-ac", "2", "-ar", rate, "-b:a", "192k", "-profile:a", "aac_low"]


def _can_carry_fields(media_info: MediaInfo) -> bool:
    """Whether a broadcast codec and raster can switch to interlaced pictures midstream."""
    return media_info.video_codec in {"mpeg2video", "h264"} and media_info.height in {
        480,
        576,
        1080,
    }


def _can_copy_video(
    media_info: MediaInfo | None,
    max_resolution: str,
    *,
    allow_upscale: bool = True,
) -> bool:
    """Can video be stream-copied (compatible codec, no processing needed)?"""
    max_h = _max_resolution_height(max_resolution) or 9999
    # SR requires re-encode (can't copy video when SR is active)
    sr_active = bool(
        allow_upscale and _sr_engine_dir is not None and _load_settings().get("sr_model", ""),
    )
    return bool(
        media_info
        and media_info.video_codec == "h264"
        and media_info.pix_fmt == "yuv420p"
        and media_info.height <= max_h
        and not sr_active
        and not media_info.interlaced,  # Can't copy if deinterlacing needed.
    )


def _can_copy_audio(media_info: MediaInfo | None, *, passthrough: bool = False) -> bool:
    """Can audio be stream-copied?"""
    if passthrough and media_info and media_info.audio_codec in PASSTHROUGH_AUDIO_CODECS:
        return True
    profile = media_info.audio_profile.upper() if media_info else ""
    return bool(
        media_info
        and media_info.audio_codec == "aac"
        and media_info.audio_sample_rate in (44_100, 48_000)
        # HE-AAC has browser compatibility issues - only copy LC-AAC.
        and "HE" not in profile
        # Surround is copied only when positively identified as LC.
        and (
            media_info.audio_channels <= 2 or (media_info.audio_channels <= 6 and profile == "LC")
        ),
    )


def _max_resolution_height(max_resolution: str) -> int | None:
    """Return the scale-down height for a resolution setting, or None for no limit."""
    return {"4k": 2160, "1080p": 1080, "720p": 720, "480p": 480}.get(
        max_resolution,
    )


def _cuttable_crop(media_info: MediaInfo | None) -> CropBox | None:
    """The detected picture area, when there are bars and a frame size to place them in."""
    if (
        media_info is None
        or media_info.crop is None
        or media_info.crop.is_full
        or not media_info.width
        or not media_info.height
    ):
        return None
    return media_info.crop
