#!/bin/bash
# Build ffmpeg from source with hardware acceleration support
# Supports: NVIDIA NVENC, AMD AMF, Intel QSV/VAAPI, LibTorch DNN, TensorRT DNN
# https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu
#
# Usage Examples:
#
#   # Default build (TensorRT enabled, requires compute_70+ GPU)
#   ./install-ffmpeg.sh
#
#   # Maxwell GPU (compute_52, e.g. GTX 900/TITAN X) - use LibTorch 2.5 instead of TensorRT.
#   # CUDA 13 dropped Maxwell: pin 12.9 (on 26.04, from NVIDIA's ubuntu2404 repo).
#   CUDA_VERSION=12.9 NVCC_GENCODE=52 ENABLE_LIBTORCH=1 ENABLE_TENSORRT=0 LIBTORCH_VERSION=2.5.0 LIBTORCH_VARIANT=cu121 ./install-ffmpeg.sh
#
#   # CPU-only DNN inference (no GPU required)
#   ENABLE_LIBTORCH=1 ENABLE_TENSORRT=0 LIBTORCH_VARIANT=cpu ./install-ffmpeg.sh
#
#   # Specific CUDA version
#   CUDA_VERSION=12.4 ./install-ffmpeg.sh
#
#   # Skip dependency installation (already installed)
#   SKIP_DEPS=1 ./install-ffmpeg.sh
#
#   # Both LibTorch and TensorRT (for testing/comparison)
#   ENABLE_LIBTORCH=1 ENABLE_TENSORRT=1 ./install-ffmpeg.sh
#
# Reruns are incremental: each library is skipped when its source revision,
# toolchain, and build settings match the last successful build (stamps in
# $BUILD_DIR/.stamps). ffmpeg itself is skipped when nothing it links changed.
#
#   # Force specific steps to rebuild (space-separated, or "all")
#   REBUILD="x265 ffmpeg" ./install-ffmpeg.sh
#
#   # Start over: wipe $BUILD_DIR and reset source trees (keeps downloads), then build
#   CLEAN=build ./install-ffmpeg.sh
#
#   # Start over from nothing: also delete $SRC_DIR (re-clones, re-downloads LibTorch/Vulkan SDK)
#   CLEAN=all ./install-ffmpeg.sh
#
#   # Clean only, no build
#   CLEAN=build PHASE=clean ./install-ffmpeg.sh
#
set -e

# =============================================================================
# Potentially Viable Pre-built FFmpeg alternatives
#
# DOCKER IMAGES
#   LinuxServer docker-ffmpeg    https://github.com/linuxserver/docker-ffmpeg
#     - Full hardware accel (NVENC, VAAPI, QSV)
#     - Comprehensive codec support, builds libva 2.23+ from source
#     - Used by Dispatcharr, basis for comparison with this script
#
# STATIC BINARIES (Linux)
#   BtbN/FFmpeg-Builds           https://github.com/BtbN/FFmpeg-Builds
#     - Daily automated builds from git master and release branches
#     - GPL/LGPL/nonfree variants, static and shared options
#     - Targets glibc 2.28+ (RHEL 8 / Ubuntu 20.04+)
#     - CUDA support: sm_52+ (Maxwell and newer)
#
#   John Van Sickle              https://johnvansickle.com/ffmpeg/
#     - Static builds for amd64, i686, armhf, arm64
#     - GPL v3 licensed, targets kernel 3.2.0+
#     - Note: static glibc = no DNS resolution (install nscd to fix)
#
# STATIC BINARIES (Windows)
#   gyan.dev                     https://www.gyan.dev/ffmpeg/builds/
#     - Essentials build: common codecs (Win 7+)
#     - Full build: all codecs including bluray, opencl (Win 10+)
#     - Official FFmpeg download page recommendation
#
# SPECIALIZED BUILDS
#   Jellyfin-ffmpeg              https://github.com/jellyfin/jellyfin-ffmpeg
#     - Modified FFmpeg with Jellyfin-specific patches
#     - Optimized for media server transcoding
#     - Ships with Jellyfin packages and Docker images
#     - Recommended only for Jellyfin; other apps should use standard builds
#
# =============================================================================

# =============================================================================
# FFmpeg library reference (checked 2026-01)
#
# Priority: high    = essential for most workflows
#           med     = useful for specific workflows
#           low     = niche use cases
#           subsumed= functionality covered by another library we use
#           legacy  = outdated, superseded by newer codecs
#
# Enable: src = built from source, apt = use apt package, - = not enabled
#
#   Library          | Build  | Pri      | Apt Ver | Latest  | Description
#   -----------------|--------|----------|---------|---------|---------------------------
#   VIDEO CODECS
#   libx264          | src    | high     | 0.164   | 0.165   | H.264/AVC encoder (8/10-bit)
#   libx265          | src    | high     | 3.5     | 4.1     | H.265/HEVC encoder (8/10/12-bit)
#   libsvtav1        | src    | high     | 1.7.0   | 3.0.2   | AV1 encoder (fast, scalable)
#   libaom           | src    | high     | 3.8.2   | 3.13.1  | AV1 reference encoder/decoder
#   libdav1d         | src    | high     | 1.4.1   | 1.5.3   | AV1 decoder (fastest)
#   libvpx           | apt    | high     | 1.14.0  | 1.14.1  | VP8/VP9 encoder/decoder
#   libvvenc         | -      | low      | -       | 1.13.1  | H.266/VVC encoder (too early)
#   librav1e         | -      | subsumed | 0.7.1   | 0.8.1   | AV1 encoder - svtav1 faster
#   libkvazaar       | -      | subsumed | 2.3.1   | 2.3.2   | HEVC encoder - x265 better
#   libopenh264      | -      | subsumed | 2.6.0   | 2.6.0   | H.264 (Cisco) - x264 better
#   libxvid          | -      | legacy   | 1.3.7   | 1.3.7   | MPEG-4 Part 2 (obsolete)
#   libtheora        | -      | legacy   | 1.2.0a1 | 1.2.0   | Theora codec (obsolete)
#
#   IMAGE CODECS
#   libwebp          | src    | high     | 1.3.2   | 1.6.0   | WebP image codec
#   libjxl           | src    | high     | 0.7.0   | 0.11.1  | JPEG XL (next-gen, HDR)
#   libopenjpeg      | -      | low      | 2.5.0   | 2.5.4   | JPEG 2000 (cinema/medical)
#   librsvg          | -      | low      | 2.58.0  | 2.61.3  | SVG rasterization
#   libsnappy        | -      | low      | 1.1.10  | 1.2.2   | Snappy compression (HAP codec)
#
#   AUDIO CODECS
#   libfdk-aac       | apt    | high     | 2.0.2   | 2.0.3   | AAC encoder (best quality)
#   libmp3lame       | apt    | high     | 3.100   | 3.100   | MP3 encoder
#   libopus          | apt    | high     | 1.5.2   | 1.6     | Opus encoder/decoder
#   libvorbis        | apt    | high     | 1.3.7   | 1.3.7   | Vorbis encoder/decoder
#   librubberband    | apt    | med      | 3.3.0   | 4.0.0   | Audio time-stretch/pitch-shift
#   liblc3           | -      | low      | 1.1.3   | 1.1.3   | LC3 Bluetooth audio codec
#   libopencore-amr  | -      | legacy   | 0.1.6   | 0.1.6   | AMR-NB/WB (old mobile audio)
#
#   SUBTITLE/TEXT
#   libass           | apt    | high     | 0.17.3  | 0.17.4  | ASS/SSA subtitle renderer
#   libfreetype      | apt    | high     | 2.13.3  | 2.14.1  | Font rendering
#   libfontconfig    | apt    | high     | 2.15.0  | 2.17.0  | Font configuration
#   libfribidi       | apt    | med      | 1.0.16  | 1.0.16  | BiDi text (RTL languages)
#   libharfbuzz      | apt    | med      | 10.2.0  | 12.3.0  | Complex text shaping
#
#   FILTERS/PROCESSING
#   libzimg          | apt    | high     | 3.0.5   | 3.0.6   | High-quality image scaling
#   libsoxr          | apt    | high     | 0.1.3   | 0.1.3   | High-quality audio resampling
#   libvmaf          | src    | med      | 2.3.1   | 3.0.0   | Video quality metrics
#   libplacebo       | src    | med      | 7.349.0 | 7.351.0 | GPU HDR tone mapping
#   libshaderc       | src*   | med      | -       | -       | GLSL->SPIRV compiler (*via Vulkan SDK)
#   libvidstab       | apt    | med      | 1.1.0   | 1.1.1   | Video stabilization
#   libmysofa        | -      | low      | 1.3.3   | 1.3.3   | HRTF spatial audio (sofalizer)
#   libtesseract     | -      | low      | 5.5.0   | 5.5.1   | OCR text extraction
#   opencl           | apt    | low      | 2.3.3   | -       | GPU compute filters
#
#   HARDWARE ACCEL
#   libva            | src    | high     | 2.20.0  | 2.23.0  | VA-API (Intel/AMD) - Xe support
#   libvpl           | src    | high     | 2023.3  | 2.16.0  | Intel QuickSync Video
#   cuda-nvcc        | src    | high     | -       | -       | NVIDIA CUDA compiler
#   nvenc            | src    | high     | -       | -       | NVIDIA hardware encoder
#   cuvid            | src    | high     | -       | -       | NVIDIA hardware decoder
#   vaapi            | src    | high     | -       | -       | VA-API hwaccel
#   nvdec            | src    | med      | -       | -       | NVIDIA hwaccel decode API
#   vulkan           | src    | med      | -       | -       | Vulkan GPU compute
#   cuda-llvm        | -      | subsumed | -       | -       | CUDA via clang - we use nvcc
#   vdpau            | -      | legacy   | 1.5     | 1.5     | NVIDIA VDPAU (use nvdec)
#
#   PROTOCOLS/NETWORK
#   openssl          | apt    | high     | 3.0.13  | 3.0.15  | TLS/HTTPS support
#   libsrt           | apt    | high     | 1.5.3   | 1.5.4   | SRT streaming protocol
#   libssh           | -      | low      | 0.10.6  | 0.11.1  | SFTP protocol
#   librist          | -      | low      | 0.2.11  | 0.2.11  | RIST broadcast protocol
#   libzmq           | -      | low      | 4.3.5   | 4.3.5   | ZeroMQ IPC messaging
#   libxml2          | -      | low      | 2.9.14  | 2.13.5  | XML/DASH manifest parsing
#
#   INPUT/OUTPUT
#   libbluray        | apt    | med      | 1.3.4   | 1.4.0   | Blu-ray disc reading
#   libv4l2          | -      | low      | 1.28.1  | 1.28.1  | V4L2 webcam/capture
#   alsa             | -      | low      | 1.2.14  | 1.2.14  | Linux ALSA audio input
#
#   META FLAGS
#   gpl              | yes    | high     | -       | -       | Enable GPL-licensed code
#   version3         | yes    | high     | -       | -       | Enable (L)GPL v3 code
#   nonfree          | yes    | high     | -       | -       | Enable non-free code (fdk-aac)
#
# =============================================================================

# Hardware acceleration (set to 1 to enable)
ENABLE_NVIDIA_CUDA=${ENABLE_NVIDIA_CUDA:-1}  # NVENC/NVDEC hardware encoding/decoding
ENABLE_AMD_AMF=${ENABLE_AMD_AMF:-1}          # AMD AMF hardware encoding (requires AMD GPU)
ENABLE_LIBTORCH=${ENABLE_LIBTORCH:-0}        # LibTorch DNN backend for AI filters (default off, prefer TensorRT)
ENABLE_TENSORRT=${ENABLE_TENSORRT:-1}        # TensorRT DNN backend for AI filters (fastest)

# LibTorch CUDA variant (only used if ENABLE_LIBTORCH=1)
# LIBTORCH_VARIANT options:
#   "cu126"   - (default) CUDA 12.6 - compatible with LibTorch 2.7+
#               Note: cu126 binaries work on CUDA 12.6+ runtimes (forward compatible)
#   "auto"    - auto-detect from CUDA_VERSION, rounding minor to nearest even
#               (PyTorch only releases cu126, cu128, cu130 - even minor versions)
#               Examples: CUDA 12.9 -> cu128, CUDA 12.7 -> cu126, CUDA 13.x -> cu130
#   "cpu"     - CPU-only (no GPU acceleration for DNN filters)
#   "cu124"   - CUDA 12.4 (for older LibTorch 2.5.x)
#   "cu126"   - force CUDA 12.6
#   "cu128"   - force CUDA 12.8
#   "cu130"   - force CUDA 13.0
#   "rocm6.4" - AMD ROCm 6.4 (requires ROCm installed on host)
LIBTORCH_VARIANT=${LIBTORCH_VARIANT:-cu126}

# Optional build components (set to 0 to use apt package instead)
BUILD_LIBPLACEBO=${BUILD_LIBPLACEBO:-1}  # GPU HDR tone mapping (requires Vulkan SDK)
BUILD_LIBX265=${BUILD_LIBX265:-1}        # H.265/HEVC encoder (apt: 3.5, latest: 4.1)
BUILD_LIBAOM=${BUILD_LIBAOM:-1}          # AV1 reference codec (apt: 3.8, latest: 3.13)
BUILD_LIBWEBP=${BUILD_LIBWEBP:-1}        # WebP image codec (apt: 1.3, latest: 1.6)
BUILD_LIBVPL=${BUILD_LIBVPL:-1}          # Intel QuickSync (apt: 2023.3, latest: 2.16)
BUILD_LIBDAV1D=${BUILD_LIBDAV1D:-1}      # AV1 decoder (apt: 1.4.1, latest: 1.5.0)
BUILD_LIBSVTAV1=${BUILD_LIBSVTAV1:-1}    # AV1 encoder (apt: 1.7.0, latest: 3.0.0)
BUILD_LIBVMAF=${BUILD_LIBVMAF:-1}        # Video quality metrics (apt: 2.3.1, latest: 3.0.0)
BUILD_LIBVA=${BUILD_LIBVA:-1}            # VA-API (apt: 2.20.0, latest: 2.23.0 - Xe support)
BUILD_LIBJXL=${BUILD_LIBJXL:-1}          # JPEG XL (apt: 0.7.0, latest: 0.11.1)
BUILD_LIBX264=${BUILD_LIBX264:-1}        # H.264 encoder (apt: 8-bit only, src: 8/10-bit)
SVTAV1_GIT_REF=${SVTAV1_GIT_REF:-}

# FFmpeg version: "snapshot" for latest git, or specific version like "7.1"
FFMPEG_VERSION=${FFMPEG_VERSION:-snapshot}

# Skip apt dependency installation (use if deps already installed, avoids sudo)
SKIP_DEPS=${SKIP_DEPS:-0}
PHASE=${PHASE:-all}            # all | deps | clean
CLEAN=${CLEAN:-}               # "" | build | all
REBUILD=${REBUILD:-}           # step names to force, or "all"

# Noninteractive apt installs (prevents prompts)
export DEBIAN_FRONTEND="${DEBIAN_FRONTEND:-noninteractive}"

# Capture script directory before any cd commands (with error handling)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || {
    echo "ERROR: Failed to determine script directory" >&2
    exit 1
}

# Validate SCRIPT_DIR is not empty and exists
if [ -z "$SCRIPT_DIR" ] || [ ! -d "$SCRIPT_DIR" ]; then
    echo "ERROR: Invalid script directory: $SCRIPT_DIR" >&2
    exit 1
fi

# =============================================================================
# Helper Functions
# =============================================================================

# Retry downloads: wget does not retry TLS handshake failures on its own,
# and one flaky connection should not abort a long build.
wget() {
    local attempt
    for attempt in 1 2 3 4 5; do
        command wget "$@" && return 0
        echo "wget failed (attempt $attempt/5), retrying in $((attempt * 10))s..." >&2
        sleep $((attempt * 10))
    done
    return 1
}

# Log error message to stderr
log_error() {
    echo "ERROR: $*" >&2
}

# Log warning message to stderr
log_warn() {
    echo "WARNING: $*" >&2
}

# Log info message
log_info() {
    echo "INFO: $*"
}

# Verify a patch was applied by checking for expected content
verify_patch() {
    local file="$1"
    local pattern="$2"
    local description="$3"
    if ! grep -q "$pattern" "$file"; then
        log_error "Patch verification failed: $description"
        log_error "Expected pattern '$pattern' not found in $file"
        return 1
    fi
    return 0
}

# Clone a git repo with validation
git_clone_validated() {
    local url="$1"
    local dir="$2"
    local depth="${3:-1}"
    local ref="${4:-}"

    if [ -n "$ref" ]; then
        git clone --depth "$depth" --branch "$ref" "$url" "$dir"
    else
        git clone --depth "$depth" "$url" "$dir"
    fi

    # Validate clone succeeded
    if [ ! -d "$dir/.git" ]; then
        log_error "Git clone failed: $url -> $dir"
        return 1
    fi
    return 0
}

# Update or clone a git repo
git_update_or_clone() {
    local url="$1"
    local dir="$2"
    local depth="${3:-1}"
    local ref="${4:-}"

    if [ -d "$dir/.git" ]; then
        # Check for conflicts or errors during pull
        local pull_output
        if ! pull_output=$(git -C "$dir" pull 2>&1); then
            log_warn "Git pull failed for $dir: $pull_output"
            log_warn "Re-cloning..."
            safe_rm_rf "$dir" 2 || rm -rf "$dir"
            git_clone_validated "$url" "$dir" "$depth" "$ref"
        elif echo "$pull_output" | grep -qE "CONFLICT|fatal"; then
            log_warn "Git pull had conflicts for $dir, re-cloning..."
            safe_rm_rf "$dir" 2 || rm -rf "$dir"
            git_clone_validated "$url" "$dir" "$depth" "$ref"
        fi
    else
        # Remove if exists but not a git repo
        if [ -e "$dir" ]; then
            safe_rm_rf "$dir" 2 || rm -rf "$dir"
        fi
        git_clone_validated "$url" "$dir" "$depth" "$ref"
    fi
}

# Detect Ubuntu version with validation
get_ubuntu_version() {
    local version_id
    if [ ! -f /etc/os-release ]; then
        log_error "/etc/os-release not found - cannot detect Ubuntu version"
        return 1
    fi
    version_id=$(grep "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)
    if [ -z "$version_id" ]; then
        log_error "Could not parse VERSION_ID from /etc/os-release"
        return 1
    fi
    # Remove dots: "24.04" -> "2404"
    echo "$version_id" | tr -d '.'
}

# Safe rm -rf: validates path before deletion to prevent catastrophic mistakes
safe_rm_rf() {
    local path="$1"
    local min_depth="${2:-2}"  # Minimum path depth (default: 2 components)

    # Never delete empty paths
    if [ -z "$path" ]; then
        log_error "safe_rm_rf: empty path"
        return 1
    fi

    # Never delete root or single-level paths
    local depth
    depth=$(echo "$path" | tr -cd '/' | wc -c)
    if [ "$depth" -lt "$min_depth" ]; then
        log_error "safe_rm_rf: path '$path' too shallow (depth $depth < $min_depth)"
        return 1
    fi

    # Never delete common dangerous paths
    case "$path" in
        /|/bin|/boot|/dev|/etc|/home|/lib|/lib64|/media|/mnt|/opt|/proc|/root|/run|/sbin|/srv|/sys|/tmp|/usr|/var)
            log_error "safe_rm_rf: refusing to delete system path: $path"
            return 1
            ;;
    esac

    # Path must exist to delete
    if [ ! -e "$path" ]; then
        return 0  # Nothing to delete
    fi

    rm -rf "$path"
}

# Validate CUDA version format (e.g., "12.4", "13.0")
validate_cuda_version() {
    local version="$1"
    if ! [[ "$version" =~ ^[0-9]+\.[0-9]+$ ]]; then
        log_error "Invalid CUDA version format: '$version' (expected X.Y)"
        return 1
    fi
    return 0
}

# Validate torch variant format (e.g., "cu124", "cu130", "cpu")
validate_torch_variant() {
    local variant="$1"
    if ! [[ "$variant" =~ ^(cu[0-9]+|rocm[0-9]+\.[0-9]+|cpu)$ ]]; then
        log_error "Invalid TORCH_VARIANT: '$variant' (expected cuXXX, rocmX.Y, or cpu)"
        return 1
    fi
    return 0
}

# Verify SHA256 checksum of a file
verify_sha256() {
    local file="$1"
    local expected="$2"

    if [ ! -f "$file" ]; then
        log_error "File not found for checksum verification: $file"
        return 1
    fi

    local actual
    actual=$(sha256sum "$file" | awk '{print $1}')

    if [ "$actual" != "$expected" ]; then
        log_error "SHA256 checksum mismatch for $file"
        log_error "  Expected: $expected"
        log_error "  Actual:   $actual"
        return 1
    fi

    log_info "SHA256 verified: $file"
    return 0
}

# Download file with optional checksum verification
download_file() {
    local url="$1"
    local output="$2"
    local sha256="${3:-}"  # Optional checksum

    log_info "Downloading: $url"
    if ! wget -q -O "$output" "$url"; then
        log_error "Failed to download: $url"
        return 1
    fi

    if [ -n "$sha256" ]; then
        if ! verify_sha256 "$output" "$sha256"; then
            rm -f "$output"
            return 1
        fi
    fi

    return 0
}

# --- Incremental build stamps ---------------------------------------------
# A step is skipped when $STAMP_DIR/<step> holds the same key as computed now.
# Key = source revision + toolchain + step-specific settings. Stamps live in
# $BUILD_DIR, so deleting it (or CLEAN=build) forces a full rebuild.
toolchain_id() {
    local os_ver=""
    [ -f /etc/os-release ] && os_ver=$(. /etc/os-release && echo "$VERSION_ID")
    echo "$os_ver|$(gcc -dumpfullversion 2>/dev/null)|$(g++ -dumpfullversion 2>/dev/null)|$(cmake --version 2>/dev/null | head -1)|$(meson --version 2>/dev/null)"
}

# step_key <src_dir> [extra settings...]
step_key() {
    local src="$1"; shift
    local rev
    rev=$(git -C "$src" rev-parse HEAD 2>/dev/null || echo "nogit")
    printf '%s\n' "$rev" "$TOOLCHAIN_ID" "$BUILD_DIR" "$@" | sha256sum | cut -c1-32
}

# step_done <name> <key> [artifact]: true if already built with this key
step_done() {
    local name="$1" key="$2" artifact="${3:-}"
    if [ "$REBUILD" = "all" ] || [[ " $REBUILD " == *" $name "* ]]; then
        return 1
    fi
    [ -n "$artifact" ] && [ ! -e "$artifact" ] && return 1
    if [ "$(cat "$STAMP_DIR/$name" 2>/dev/null)" = "$key" ]; then
        log_info "$name: up to date, skipping (REBUILD=\"$name\" to force)"
        return 0
    fi
    return 1
}

step_mark() {
    mkdir -p "$STAMP_DIR"
    echo "$2" > "$STAMP_DIR/$1"
}

# trt_pick_version <pkg> <cuda X-Y>: TensorRT 10.x version string of <pkg> built for this
# CUDA major: the newest whose +cudaA.B tag is <= X.Y, else the newest with the lowest tag.
trt_pick_version() {
    local pkg="$1" cuda="${2//-/.}" major="${2%%[-.]*}"
    local versions
    versions=$(apt-cache madison "$pkg" 2>/dev/null | awk '{print $3}' | grep -E "^10\..*\+cuda${major}\.[0-9]+$" | sort -uV)
    [ -n "$versions" ] || return 0
    local v pick="" low_tag low_pick=""
    low_tag=$(sed 's/.*+cuda//' <<< "$versions" | sort -V | head -1)
    while read -r v; do
        version_ge "$cuda" "${v##*+cuda}" && pick="$v"
        [ "${v##*+cuda}" = "$low_tag" ] && low_pick="$v"
    done <<< "$versions"
    echo "${pick:-$low_pick}"
}

version_ge() {
    [ "$(printf '%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]
}

ensure_meson_min_version() {
    local min_version="$1"
    local current_version

    current_version=$(meson --version 2>/dev/null || echo 0)
    if ! version_ge "$current_version" "$min_version"; then
        echo "Meson $current_version < $min_version, upgrading via pip..."
        sudo apt-get update
        sudo apt-get install -y python3-pip
        if pip3 install --help 2>/dev/null | grep -q -- '--break-system-packages'; then
            sudo -E pip3 install --upgrade --break-system-packages meson
        else
            sudo -E pip3 install --upgrade meson
        fi
        export PATH="/usr/local/bin:$PATH"
    fi
}

# NVIDIA CUDA setup (only used if ENABLE_NVIDIA_CUDA=1)
# CUDA_VERSION options:
#   "auto"    - (default) use installed CUDA if available, else install latest
#   "12.8"    - explicit version (e.g., 12.4, 12.6, 13.0) - dots converted to dashes internally
CUDA_VERSION=${CUDA_VERSION:-auto}
# Validate CUDA_VERSION format before normalization
if [ "$CUDA_VERSION" != "auto" ]; then
    if ! validate_cuda_version "$CUDA_VERSION"; then
        log_error "Set CUDA_VERSION to 'auto' or a valid version like '12.8' or '13.0'"
        exit 1
    fi
fi
# Normalize: convert "12.4" to "12-4" for apt package names
CUDA_VERSION="${CUDA_VERSION//./-}"
# NVCC_GENCODE options:
#   "native"  - (default) compile for build machine's GPU via nvidia-smi
#   "minimum" - lowest arch for CUDA version (sm_52 for <13, sm_75 for 13+)
#   "75"      - explicit single arch (e.g., 75, 86, 89)
NVCC_GENCODE=${NVCC_GENCODE:-native}

# Build paths
SRC_DIR="${SRC_DIR:-$HOME/ffmpeg_sources}"      # Source code cache (can be deleted after build)
BUILD_DIR="${BUILD_DIR:-$HOME/ffmpeg_build}"    # Build artifacts cache (can be deleted after build)
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"          # Final binary install location
LIB_DIR="${LIB_DIR:-$HOME/.local/lib}"          # Final shared library install location (for libva)
STAMP_DIR="$BUILD_DIR/.stamps"

# Get number of processors with fallback and validation
NPROC=$(nproc 2>/dev/null || echo 4)
if ! [[ "$NPROC" =~ ^[0-9]+$ ]] || [ "$NPROC" -lt 1 ] || [ "$NPROC" -gt 256 ]; then
    log_warn "Invalid NPROC value '$NPROC', using 4"
    NPROC=4
fi

# Ensure HOME is set (for cron/systemd contexts)
if [ -z "$HOME" ]; then
    HOME=$(getent passwd "$(id -un)" 2>/dev/null | cut -d: -f6) || true
    if [ -z "$HOME" ] || [ ! -d "$HOME" ]; then
        log_error "HOME environment variable not set and could not be detected"
        exit 1
    fi
    export HOME
fi

case "$CLEAN" in
    "") ;;
    build)
        echo "CLEAN=build: removing $BUILD_DIR and resetting source trees in $SRC_DIR"
        safe_rm_rf "$BUILD_DIR" 3
        if [ -d "$SRC_DIR" ]; then
            for d in "$SRC_DIR"/*/; do
                d="${d%/}"
                if [ -d "$d/.git" ]; then
                    git -C "$d" reset -q --hard
                    git -C "$d" clean -q -ffdx
                    git -C "$d" submodule foreach -q --recursive 'git clean -q -ffdx' 2>/dev/null || true
                fi
            done
            # Extracted ffmpeg tarball trees carry in-place patches; re-extract fresh
            for d in "$SRC_DIR"/ffmpeg-*/; do
                [ -d "$d" ] && safe_rm_rf "${d%/}" 3
            done
            safe_rm_rf "$SRC_DIR/aom_build" 3
        fi
        ;;
    all)
        echo "CLEAN=all: removing $BUILD_DIR and $SRC_DIR"
        safe_rm_rf "$BUILD_DIR" 3
        safe_rm_rf "$SRC_DIR" 3
        ;;
    *)
        log_error "Invalid CLEAN='$CLEAN' (expected build or all)"
        exit 1
        ;;
esac
if [ "$PHASE" = "clean" ]; then
    echo "PHASE=clean set; done."
    exit 0
fi

# Create build directories (with validation)
mkdir -p "$SRC_DIR" "$BUILD_DIR" "$BIN_DIR" "$LIB_DIR" || {
    log_error "Failed to create build directories"
    exit 1
}

# Note: libplacebo pin was previously needed for jammy because older versions
# lacked dependencies. Now using latest which is compatible with FFmpeg snapshot API.

# Note: SVT-AV1 pin was previously needed for FFmpeg 7.0 API compatibility on jammy.
# With FFmpeg snapshot, we use latest SVT-AV1 (no pin needed).

# Base packages (installed first, includes wget needed for CUDA repo setup)
APT_PACKAGES=(
    autoconf
    automake
    build-essential
    cmake
    doxygen
    git
    meson
    nasm
    ninja-build
    pkg-config
    texinfo
    unzip
    wget
    xxd
    yasm
    libass-dev
    libbluray-dev
    libfdk-aac-dev
    libfontconfig1-dev
    libfreetype6-dev
    libfribidi-dev
    libharfbuzz-dev
    libsoxr-dev
    libsrt-openssl-dev
    libssl-dev
    libzstd-dev
    libzimg-dev
    liblzma-dev
    liblzo2-dev
    libmp3lame-dev
    libnuma-dev
    ocl-icd-opencl-dev
    libopus-dev
    librubberband-dev
    libsdl2-dev
    libtool
    python3-jinja2
    libunistring-dev
    libvdpau-dev
    libvidstab-dev
    libdrm-dev
    libx11-dev
    libvorbis-dev
    libvpx-dev
    libxcb-shm0-dev
    libxcb-xfixes0-dev
    libxcb1-dev
    zlib1g-dev
    # Intel oneVPL/QSV runtime (needed for Intel GPU hardware encoding)
    libmfx-gen1.2
)
# Ubuntu 26.04+: libcrypto.pc's static libs reference libjitterentropy.a (needed to link srt/openssl)
if apt-cache show libjitterentropy3-dev >/dev/null 2>&1; then
    APT_PACKAGES+=(libjitterentropy3-dev)
fi
# Add apt packages for libraries we're not building from source
[ "$BUILD_LIBX265" != "1" ] && APT_PACKAGES+=(libx265-dev)
[ "$BUILD_LIBAOM" != "1" ] && APT_PACKAGES+=(libaom-dev)
[ "$BUILD_LIBWEBP" != "1" ] && APT_PACKAGES+=(libwebp-dev)
[ "$BUILD_LIBVPL" != "1" ] && APT_PACKAGES+=(libvpl-dev)
[ "$BUILD_LIBDAV1D" != "1" ] && APT_PACKAGES+=(libdav1d-dev)
[ "$BUILD_LIBSVTAV1" != "1" ] && APT_PACKAGES+=(libsvtav1enc-dev)
[ "$BUILD_LIBVMAF" != "1" ] && APT_PACKAGES+=(libvmaf-dev)
[ "$BUILD_LIBVA" != "1" ] && APT_PACKAGES+=(libva-dev)
[ "$BUILD_LIBJXL" != "1" ] && APT_PACKAGES+=(libjxl-dev)
[ "$BUILD_LIBX264" != "1" ] && APT_PACKAGES+=(libx264-dev)
# Note: TensorRT headers (libnvinfer-headers-dev) installed later after CUDA repo is set up
if [ "$SKIP_DEPS" != "1" ]; then
    sudo apt-get update
    sudo apt-get install -y "${APT_PACKAGES[@]}"
    ensure_meson_min_version 0.63
fi


CUDA_FLAGS=()
NVCC_ARCH=""

if [ "$ENABLE_NVIDIA_CUDA" = "1" ]; then
    # Check if CUDA is already installed
    if [ "$CUDA_VERSION" = "auto" ]; then
        if command -v nvcc &> /dev/null; then
            # Extract version from nvcc (e.g., "12.9" -> "12-9")
            NVCC_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
            CUDA_VERSION=$(echo "$NVCC_VERSION" | tr '.' '-')
            echo "Detected installed CUDA $NVCC_VERSION (using version $CUDA_VERSION)"
        else
            echo "No CUDA installed, will install latest from NVIDIA repo"
        fi
    fi

    # Add CUDA repo if not present or if we need to install
    if [ "$SKIP_DEPS" != "1" ]; then
        if [ "$CUDA_VERSION" = "auto" ] || ! command -v nvcc &> /dev/null; then
            if ! dpkg -l cuda-keyring 2>/dev/null | grep -q ^ii; then
                # Detect Ubuntu version for correct CUDA repo (24.04 -> ubuntu2404, 25.04 -> ubuntu2504)
                UBUNTU_VERSION=$(get_ubuntu_version) || exit 1
                CUDA_REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/cuda-keyring_1.1-1_all.deb"
                CUDA_KEYRING_DEB="$(mktemp -d)/cuda-keyring.deb"
                if ! wget --progress=dot:giga "$CUDA_REPO_URL" -O "$CUDA_KEYRING_DEB"; then
                    log_error "Failed to download CUDA keyring from $CUDA_REPO_URL"
                    exit 1
                fi
                sudo dpkg -i "$CUDA_KEYRING_DEB"
                rm -rf "$(dirname "$CUDA_KEYRING_DEB")"
                sudo apt-get update
            fi

            # Get latest CUDA version if still auto
            if [ "$CUDA_VERSION" = "auto" ]; then
                CUDA_VERSION=$(apt-cache search '^cuda-nvcc-[0-9]' | sed 's/cuda-nvcc-//' | cut -d' ' -f1 | sort -V | tail -1)
                if [ -z "$CUDA_VERSION" ]; then
                    echo "Error: No CUDA packages found. Install CUDA repo first or set CUDA_VERSION manually." >&2
                    exit 1
                fi
                echo "Will install latest CUDA version: $CUDA_VERSION"
            fi
        fi

        # Each NVIDIA repo carries only the toolkits current for its release: ubuntu2604 has
        # CUDA 13+, ubuntu2404 has 12.5+, ubuntu2204 still has 12.4. Older toolkits install
        # fine on newer Ubuntu (self-contained under /usr/local/cuda-X.Y), so walk back
        # through older repos until the requested version is found.
        cuda_pkg_available() {
            apt-cache policy "cuda-nvcc-$CUDA_VERSION" 2>/dev/null | awk '/Candidate:/{print $2}' | grep -qv '(none)'
        }
        for CUDA_FALLBACK_REL in 2404 2204; do
            cuda_pkg_available && break
            CUDA_FALLBACK_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${CUDA_FALLBACK_REL}/x86_64/"
            # Already configured (e.g. native repo on that release): adding it again would
            # conflict on Signed-By.
            if grep -rqs "repos/ubuntu${CUDA_FALLBACK_REL}/x86_64" /etc/apt/sources.list /etc/apt/sources.list.d/; then
                continue
            fi
            echo "cuda-nvcc-$CUDA_VERSION not found; adding NVIDIA ubuntu${CUDA_FALLBACK_REL} repo"
            # Older repos are signed with a different key than newer ones; install side by side
            CUDA_FALLBACK_KEY="/usr/share/keyrings/cuda-archive-keyring-ubuntu${CUDA_FALLBACK_REL}.gpg"
            if [ ! -f "$CUDA_FALLBACK_KEY" ]; then
                CUDA_KEY_TMP=$(mktemp -d)
                if ! wget -q -O "$CUDA_KEY_TMP/k.deb" "${CUDA_FALLBACK_URL}cuda-keyring_1.1-1_all.deb"; then
                    log_error "Failed to download ubuntu${CUDA_FALLBACK_REL} CUDA keyring"
                    exit 1
                fi
                dpkg-deb -x "$CUDA_KEY_TMP/k.deb" "$CUDA_KEY_TMP/x"
                sudo install -m 644 "$CUDA_KEY_TMP/x/usr/share/keyrings/cuda-archive-keyring.gpg" "$CUDA_FALLBACK_KEY"
                rm -rf "$CUDA_KEY_TMP"
            fi
            sudo tee "/etc/apt/sources.list.d/cuda-ubuntu${CUDA_FALLBACK_REL}-x86_64.sources" >/dev/null <<SRCEOF
Types: deb
URIs: $CUDA_FALLBACK_URL
Suites: /
Signed-By: $CUDA_FALLBACK_KEY
SRCEOF
            sudo apt-get update
        done
        if ! cuda_pkg_available; then
            log_error "cuda-nvcc-$CUDA_VERSION not available from NVIDIA's repos"
            exit 1
        fi

        # Install CUDA packages
        sudo apt-get install -y libffmpeg-nvenc-dev cuda-nvcc-$CUDA_VERSION cuda-cudart-dev-$CUDA_VERSION

        # Install TensorRT headers only (requires NVIDIA repo set up above)
        # We only need headers for compilation - libnvinfer is loaded via dlopen at runtime.
        # The backend dlopens libnvinfer.so.10, so headers must be TensorRT 10, built for
        # this CUDA (newest build tagged <= this CUDA in the same major; else oldest of it).
        if [ "$ENABLE_TENSORRT" = "1" ]; then
            TRT_VERSION=$(trt_pick_version libnvinfer-headers-dev "$CUDA_VERSION")
            if [ -z "$TRT_VERSION" ]; then
                log_error "No TensorRT 10 headers for CUDA $CUDA_VERSION in NVIDIA's repos"
                exit 1
            fi
            echo "TensorRT headers: $TRT_VERSION"
            sudo apt-get install -y --allow-downgrades "libnvinfer-headers-dev=$TRT_VERSION"
        fi
    fi
    echo "Using CUDA version: $CUDA_VERSION"

    # Detect CUDA installation path
    CUDA_VERSION_DOT=$(echo "$CUDA_VERSION" | tr '-' '.')
    if [ -x "/usr/local/cuda-${CUDA_VERSION_DOT}/bin/nvcc" ]; then
        CUDA_PATH="/usr/local/cuda-${CUDA_VERSION_DOT}"
    elif [ -d "/usr/local/cuda" ]; then
        CUDA_PATH="/usr/local/cuda"
    else
        echo "Warning: CUDA path not found, using /usr/local/cuda (headers may be missing)" >&2
        CUDA_PATH="/usr/local/cuda"
    fi
    echo "Using CUDA path: $CUDA_PATH"

    # Patch CUDA headers for glibc 2.41+ (Ubuntu 25.04+). glibc now declares C23
    # sinpi/cospi/rsqrt (+f) as noexcept; CUDA 12.x declares them without it, so nvcc
    # fails with "exception specification is incompatible". Add matching noexcept.
    CUDA_MATH_HEADER="$CUDA_PATH/targets/x86_64-linux/include/crt/math_functions.h"
    GLIBC_VERSION=$(ldd --version | head -1 | grep -oP '\d+\.\d+$')
    if [ -f "$CUDA_MATH_HEADER" ] && version_ge "$GLIBC_VERSION" 2.41; then
        C23_DECL_RE='^(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ (double|float) +(sinpif?|cospif?|rsqrtf?)\((double|float) x\));'
        if grep -qE "$C23_DECL_RE\$" "$CUDA_MATH_HEADER"; then
            if [ "$SKIP_DEPS" = "1" ]; then
                log_warn "CUDA headers need a glibc $GLIBC_VERSION patch (requires sudo); rerun without SKIP_DEPS=1"
            else
                echo "Patching CUDA headers for glibc $GLIBC_VERSION compatibility..."
                [ -f "${CUDA_MATH_HEADER}.orig" ] || sudo cp "$CUDA_MATH_HEADER" "${CUDA_MATH_HEADER}.orig"
                sudo sed -i -E "s/${C23_DECL_RE}\$/\1 noexcept (true);/" "$CUDA_MATH_HEADER"
                if grep -qE "$C23_DECL_RE\$" "$CUDA_MATH_HEADER"; then
                    log_error "CUDA header patch failed: $CUDA_MATH_HEADER"
                    exit 1
                fi
            fi
        fi
    fi

    # Pick a host compiler nvcc accepts. CUDA 12.x rejects GCC newer than 14
    # (Ubuntu 26.04 ships 15); fall back to an older g++, installing it if needed.
    if [ -z "${NVCC_CCBIN:-}" ]; then
        NVCC_PROBE_DIR=$(mktemp -d)
        printf '__global__ void k(){}\n' > "$NVCC_PROBE_DIR/p.cu"
        nvcc_accepts() {
            if [ -n "$1" ]; then set -- env NVCC_CCBIN="$1"; else set -- env -u NVCC_CCBIN; fi
            "$@" "$CUDA_PATH/bin/nvcc" -std=c++14 -c -o "$NVCC_PROBE_DIR/p.o" "$NVCC_PROBE_DIR/p.cu" \
                -Wno-deprecated-gpu-targets >/dev/null 2>&1
        }
        if ! nvcc_accepts ""; then
            for v in 14 13 12; do
                if ! command -v "g++-$v" >/dev/null && [ "$SKIP_DEPS" != "1" ]; then
                    sudo apt-get install -y "g++-$v" >/dev/null 2>&1 || continue
                fi
                if command -v "g++-$v" >/dev/null && nvcc_accepts "g++-$v"; then
                    export NVCC_CCBIN="g++-$v"
                    break
                fi
            done
            if [ -z "${NVCC_CCBIN:-}" ]; then
                log_error "nvcc ($CUDA_PATH) accepts no available host compiler; set NVCC_CCBIN"
                exit 1
            fi
        fi
        rm -rf "$NVCC_PROBE_DIR"
    fi
    echo "nvcc host compiler: ${NVCC_CCBIN:-default ($(g++ -dumpfullversion))}"

    CUDA_FLAGS=(--enable-cuda-nvcc --enable-nvenc --enable-cuvid --enable-nvdec)

    CUDA_MAJOR="${CUDA_VERSION%%-*}"

    if [ "$NVCC_GENCODE" = "native" ]; then
        # Detect GPU compute capability
        if ! command -v nvidia-smi &> /dev/null; then
            log_warn "nvidia-smi not found, falling back to minimum arch"
            NVCC_GENCODE="minimum"
        elif ! COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1); then
            log_warn "nvidia-smi failed to query GPU, falling back to minimum arch"
            NVCC_GENCODE="minimum"
        elif [ -z "$COMPUTE_CAP" ]; then
            log_warn "nvidia-smi found but no GPU detected, falling back to minimum"
            NVCC_GENCODE="minimum"
        else
            COMPUTE_CAP_NUM=$(echo "$COMPUTE_CAP" | tr -d '.')
            NVCC_ARCH="-arch=sm_${COMPUTE_CAP_NUM}"
            log_info "CUDA $CUDA_VERSION NVCC_GENCODE=native -> $NVCC_ARCH (detected via nvidia-smi)"
        fi
    fi

    if [ "$NVCC_GENCODE" = "minimum" ]; then
        if [ "$CUDA_MAJOR" -ge 13 ]; then
            NVCC_ARCH="-arch=sm_75"
        else
            NVCC_ARCH="-arch=sm_52"
        fi
        echo "CUDA $CUDA_VERSION NVCC_GENCODE=minimum -> $NVCC_ARCH"
    elif [ "$NVCC_GENCODE" != "native" ]; then
        # Explicit arch number
        NVCC_ARCH="-arch=sm_$NVCC_GENCODE"
        echo "CUDA $CUDA_VERSION NVCC_GENCODE=$NVCC_GENCODE -> $NVCC_ARCH"
    fi

    # nv-codec-headers sets the NVENC/NVDEC API ffmpeg requires at runtime; the driver
    # must support it or every nvenc open fails ("Driver does not support the required
    # nvenc API version"). Pick the newest branch the installed driver supports
    # (minimums from each branch's README). Override with NV_CODEC_REF=<branch>.
    if [ -z "${NV_CODEC_REF:-}" ]; then
        NV_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        if [ "${FFMPEG_IMAGE:-}" = "netv-ffmpeg:cuda12.4" ] || [ "${CUDA_VERSION:-}" = "12-4" ]; then
            NV_CODEC_REF="sdk/12.2"
        elif [ -z "$NV_DRIVER" ]; then
            # No driver to probe (e.g. image builds): target the minimum driver this CUDA
            # needs anyway, so old-hardware images keep working NVENC.
            # CUDA <12.8 runs on 550+ drivers (sdk/12.2); 12.8+ needs 570+ (sdk/13.0).
            if version_ge "${CUDA_VERSION//-/.}" 12.8; then
                NV_CODEC_REF="sdk/13.0"
            else
                NV_CODEC_REF="sdk/12.2"
            fi
        elif version_ge "$NV_DRIVER" 610.0; then
            NV_CODEC_REF="master"
        elif version_ge "$NV_DRIVER" 570.0; then
            NV_CODEC_REF="sdk/13.0"
        elif version_ge "$NV_DRIVER" 550.54.14; then
            NV_CODEC_REF="sdk/12.2"
        elif version_ge "$NV_DRIVER" 530.41.03; then
            NV_CODEC_REF="sdk/12.1"
        else
            NV_CODEC_REF="sdk/12.0"
        fi
    fi
    echo "nv-codec-headers: driver=${NV_DRIVER:-none}, ref=$NV_CODEC_REF"

    cd "$SRC_DIR"
    if [ -d nv-codec-headers/.git ]; then
        git -C nv-codec-headers fetch --depth 1 origin "$NV_CODEC_REF"
        git -C nv-codec-headers checkout -q -f FETCH_HEAD
    else
        git clone --depth 1 --branch "$NV_CODEC_REF" https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
    fi
    cd nv-codec-headers
    make
    make PREFIX="$BUILD_DIR" install
fi


# AMD AMF setup (hardware encoding for AMD GPUs)
# AMF is header-only at build time; runtime driver comes from host's AMD GPU driver
AMF_FLAGS=()
if [ "$ENABLE_AMD_AMF" = "1" ]; then
    cd "$SRC_DIR"
    git_update_or_clone "https://github.com/GPUOpen-LibrariesAndSDKs/AMF.git" "AMF" 1
    if [ ! -d "AMF/amf/public/include" ]; then
        log_error "AMF clone succeeded but include directory not found"
        exit 1
    fi
    mkdir -p "$BUILD_DIR/include/AMF"
    # Use /. to copy directory contents without glob expansion issues
    cp -r "AMF/amf/public/include/." "$BUILD_DIR/include/AMF/"
    AMF_FLAGS=(--enable-amf)
    log_info "AMF headers installed for AMD GPU encoding"
fi

if [ "$PHASE" = "deps" ]; then
    echo "PHASE=deps set; skipping source builds."
    exit 0
fi


# libx264 (H.264/AVC encoder)
# Build with --bit-depth=all for 8-bit and 10-bit support
TOOLCHAIN_ID=$(toolchain_id)

if [ "$BUILD_LIBX264" = "1" ]; then
    cd "$SRC_DIR"
    git_update_or_clone "https://code.videolan.org/videolan/x264.git" "x264" 1
    KEY=$(step_key "$SRC_DIR/x264")
    if ! step_done x264 "$KEY" "$BUILD_DIR/lib/libx264.a"; then
        cd x264
        PATH="$BIN_DIR:$PATH" ./configure --prefix="$BUILD_DIR" --enable-static --enable-pic --disable-cli --bit-depth=all
        PATH="$BIN_DIR:$PATH" make -j "$NPROC"
        make install
        step_mark x264 "$KEY"
    fi
fi


# libx265 (H.265/HEVC encoder)
# Multilib build: 8-bit + 10-bit + 12-bit support (required for HDR)
# Build order: 12-bit → 10-bit → 8-bit (main links the others)
if [ "$BUILD_LIBX265" = "1" ]; then
    cd "$SRC_DIR"
    git_update_or_clone "https://bitbucket.org/multicoreware/x265_git.git" "x265_git" 1
    KEY=$(step_key "$SRC_DIR/x265_git")
fi
if [ "$BUILD_LIBX265" = "1" ] && ! step_done x265 "$KEY" "$BUILD_DIR/lib/libx265.a"; then
    cd "$SRC_DIR/x265_git/build/linux"

    # Clean previous builds
    rm -rf 8bit 10bit 12bit
    mkdir -p 8bit 10bit 12bit

    # Build 12-bit
    cd 12bit
    PATH="$BIN_DIR:$PATH" cmake -G "Unix Makefiles" \
        -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" \
        -DHIGH_BIT_DEPTH=ON \
        -DEXPORT_C_API=OFF \
        -DENABLE_SHARED=OFF \
        -DENABLE_CLI=OFF \
        -DMAIN12=ON \
        ../../../source
    PATH="$BIN_DIR:$PATH" make -j "$NPROC"

    # Build 10-bit
    cd ../10bit
    PATH="$BIN_DIR:$PATH" cmake -G "Unix Makefiles" \
        -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" \
        -DHIGH_BIT_DEPTH=ON \
        -DEXPORT_C_API=OFF \
        -DENABLE_SHARED=OFF \
        -DENABLE_CLI=OFF \
        ../../../source
    PATH="$BIN_DIR:$PATH" make -j "$NPROC"

    # Build 8-bit (main) and link in 10-bit and 12-bit
    cd ../8bit
    ln -sf ../10bit/libx265.a libx265_main10.a
    ln -sf ../12bit/libx265.a libx265_main12.a
    PATH="$BIN_DIR:$PATH" cmake -G "Unix Makefiles" \
        -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" \
        -DLIB_INSTALL_DIR="$BUILD_DIR/lib" \
        -DENABLE_SHARED=OFF \
        -DENABLE_CLI=OFF \
        -DEXTRA_LIB="x265_main10.a;x265_main12.a" \
        -DEXTRA_LINK_FLAGS="-L." \
        -DLINKED_10BIT=ON \
        -DLINKED_12BIT=ON \
        ../../../source
    PATH="$BIN_DIR:$PATH" make -j "$NPROC"
    # Merge 8-bit, 10-bit, and 12-bit libraries into one (cmake doesn't do this automatically)
    mv libx265.a libx265_main.a
    mkdir -p merged/8bit merged/10bit merged/12bit
    (cd merged/8bit && ar x ../../libx265_main.a)
    (cd merged/10bit && ar x ../../libx265_main10.a)
    (cd merged/12bit && ar x ../../libx265_main12.a)
    ar crs libx265.a merged/*/*.o
    rm -rf merged libx265_main.a
    make install

    # x265's cmake doesn't reliably install x265.pc, so we create it manually
    # Extract version from x265.h (format: #define X265_BUILD 215)
    X265_VERSION=$(grep '#define X265_BUILD' "$BUILD_DIR/include/x265.h" | awk '{print $3}')
    mkdir -p "$BUILD_DIR/lib/pkgconfig"
    cat > "$BUILD_DIR/lib/pkgconfig/x265.pc" << PCEOF
prefix=$BUILD_DIR
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib
includedir=\${prefix}/include

Name: x265
Description: H.265/HEVC video encoder (8-bit + 10-bit + 12-bit)
Version: $X265_VERSION
Libs: -L\${libdir} -lx265
Libs.private: -lstdc++ -lm -lrt -ldl -lnuma -lpthread
Cflags: -I\${includedir}
PCEOF
    step_mark x265 "$KEY"
fi

# libaom (AV1 reference codec)
if [ "$BUILD_LIBAOM" = "1" ]; then
    cd "$SRC_DIR"
    git_update_or_clone "https://aomedia.googlesource.com/aom" "aom" 1
    KEY=$(step_key "$SRC_DIR/aom")
    if ! step_done aom "$KEY" "$BUILD_DIR/lib/libaom.a"; then
        mkdir -p aom_build
        cd aom_build
        PATH="$BIN_DIR:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" -DENABLE_TESTS=OFF -DENABLE_NASM=on -DBUILD_SHARED_LIBS=OFF -DCONFIG_AV1_HIGHBITDEPTH=1 ../aom
        PATH="$BIN_DIR:$PATH" make -j "$NPROC"
        make install
        step_mark aom "$KEY"
    fi
fi

# libwebp (WebP image codec)
if [ "$BUILD_LIBWEBP" = "1" ]; then
    cd "$SRC_DIR"
    git_update_or_clone "https://chromium.googlesource.com/webm/libwebp" "libwebp" 1
    KEY=$(step_key "$SRC_DIR/libwebp")
    if ! step_done libwebp "$KEY" "$BUILD_DIR/lib/libwebp.a"; then
        cd libwebp
        ./autogen.sh
        ./configure --prefix="$BUILD_DIR" --disable-shared --enable-static
        make -j "$NPROC"
        make install
        step_mark libwebp "$KEY"
    fi
fi

# libjxl (JPEG XL image codec)
# Ubuntu 24.04 ships 0.7.0 which is quite old; latest is 0.11.1 with HDR improvements
if [ "$BUILD_LIBJXL" = "1" ]; then
    cd "$SRC_DIR"
    # libjxl needs --recursive for submodules
    if [ -d libjxl/.git ]; then
        git -C libjxl pull
        git -C libjxl submodule update --init --recursive
    else
        rm -rf libjxl
        git clone --depth 1 --recursive https://github.com/libjxl/libjxl.git
        if [ ! -d libjxl/.git ]; then
            log_error "libjxl clone failed"
            exit 1
        fi
    fi
    KEY=$(step_key "$SRC_DIR/libjxl" "$(git -C "$SRC_DIR/libjxl" submodule status --recursive 2>/dev/null)")
    if ! step_done libjxl "$KEY" "$BUILD_DIR/lib/libjxl.a"; then
        cd libjxl
        mkdir -p build
        cd build
        cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_SHARED_LIBS=OFF -DJPEGXL_ENABLE_BENCHMARK=OFF -DJPEGXL_ENABLE_EXAMPLES=OFF \
            -DJPEGXL_ENABLE_MANPAGES=OFF -DJPEGXL_ENABLE_PLUGINS=OFF -DJPEGXL_ENABLE_VIEWERS=OFF \
            -DJPEGXL_ENABLE_TOOLS=OFF -DJPEGXL_ENABLE_DOXYGEN=OFF -DJPEGXL_ENABLE_JPEGLI=OFF ..
        make -j "$NPROC"
        make install
        step_mark libjxl "$KEY"
    fi
fi

# libvpl (Intel Video Processing Library / QuickSync)
if [ "$BUILD_LIBVPL" = "1" ]; then
    cd "$SRC_DIR"
    git_update_or_clone "https://github.com/intel/libvpl.git" "libvpl" 1
    KEY=$(step_key "$SRC_DIR/libvpl")
    if ! step_done libvpl "$KEY" "$BUILD_DIR/lib/libvpl.a"; then
        mkdir -p libvpl/build
        cd libvpl/build
        cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" -DBUILD_SHARED_LIBS=OFF ..
        make -j "$NPROC"
        make install
        step_mark libvpl "$KEY"
    fi
fi

# libdav1d (AV1 decoder)
if [ "$BUILD_LIBDAV1D" = "1" ]; then
    cd "$SRC_DIR"
    git_update_or_clone "https://code.videolan.org/videolan/dav1d.git" "dav1d" 1
    KEY=$(step_key "$SRC_DIR/dav1d")
    if ! step_done dav1d "$KEY" "$BUILD_DIR/lib/libdav1d.a"; then
        cd dav1d
        if [ -f build/build.ninja ]; then
            meson setup --reconfigure build --buildtype=release --default-library=static --prefix="$BUILD_DIR" --libdir="$BUILD_DIR/lib" || \
            meson setup --wipe build --buildtype=release --default-library=static --prefix="$BUILD_DIR" --libdir="$BUILD_DIR/lib"
        else
            meson setup build --buildtype=release --default-library=static --prefix="$BUILD_DIR" --libdir="$BUILD_DIR/lib"
        fi
        ninja -C build
        ninja -C build install
        step_mark dav1d "$KEY"
    fi
fi

# libsvtav1 (AV1 encoder)
if [ "$BUILD_LIBSVTAV1" = "1" ]; then
    cd "$SRC_DIR"
    git_update_or_clone "https://gitlab.com/AOMediaCodec/SVT-AV1.git" "SVT-AV1" 1
    mkdir -p SVT-AV1/build
    cd SVT-AV1/build
    if [ -n "$SVTAV1_GIT_REF" ]; then
        git -C .. fetch --depth 1 origin "$SVTAV1_GIT_REF"
        git -C .. checkout -q FETCH_HEAD
    fi
    KEY=$(step_key "$SRC_DIR/SVT-AV1")
    if ! step_done svtav1 "$KEY" "$BUILD_DIR/lib/libSvtAv1Enc.a"; then
        PATH="$BIN_DIR:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DBUILD_DEC=OFF -DBUILD_SHARED_LIBS=OFF ..
        PATH="$BIN_DIR:$PATH" make -j "$NPROC"
        make install
        step_mark svtav1 "$KEY"
    fi
fi

# libvmaf (video quality metrics)
if [ "$BUILD_LIBVMAF" = "1" ]; then
    cd "$SRC_DIR"
    git_update_or_clone "https://github.com/Netflix/vmaf" "vmaf" 1
    KEY=$(step_key "$SRC_DIR/vmaf" "$BIN_DIR")
    if ! step_done vmaf "$KEY" "$BUILD_DIR/lib/libvmaf.a"; then
        mkdir -p vmaf/libvmaf/build
        cd vmaf/libvmaf/build
        if [ -f build.ninja ]; then
            meson setup --reconfigure -Denable_tests=false -Denable_docs=false --buildtype=release --default-library=static '../' --prefix "$BUILD_DIR" --bindir="$BIN_DIR" --libdir="$BUILD_DIR/lib" || \
            meson setup --wipe -Denable_tests=false -Denable_docs=false --buildtype=release --default-library=static '../' --prefix "$BUILD_DIR" --bindir="$BIN_DIR" --libdir="$BUILD_DIR/lib"
        else
            meson setup -Denable_tests=false -Denable_docs=false --buildtype=release --default-library=static '../' --prefix "$BUILD_DIR" --bindir="$BIN_DIR" --libdir="$BUILD_DIR/lib"
        fi
        ninja
        ninja install
        step_mark vmaf "$KEY"
    fi
fi

# libva (VA-API)
# Ubuntu 24.04 ships 2.20.0 which lacks Intel Xe kernel driver support (added in 2.21)
# Build from source to get Xe support for newer Intel GPUs
if [ "$BUILD_LIBVA" = "1" ]; then
    cd "$SRC_DIR"
    git_update_or_clone "https://github.com/intel/libva.git" "libva" 1
    KEY=$(step_key "$SRC_DIR/libva")
    if ! step_done libva "$KEY" "$BUILD_DIR/lib/libva.so"; then
        cd libva
        if [ -f build/build.ninja ]; then
            meson setup --reconfigure build --buildtype=release --default-library=shared --prefix="$BUILD_DIR" --libdir="$BUILD_DIR/lib" || \
            meson setup --wipe build --buildtype=release --default-library=shared --prefix="$BUILD_DIR" --libdir="$BUILD_DIR/lib"
        else
            meson setup build --buildtype=release --default-library=shared --prefix="$BUILD_DIR" --libdir="$BUILD_DIR/lib"
        fi
        ninja -C build
        ninja -C build install
        step_mark libva "$KEY"
    fi
    # Copy shared libs to permanent location (LIB_DIR) for runtime
    cp -a "$BUILD_DIR/lib"/libva*.so* "$LIB_DIR/"
fi

meson_supports_prefer_static() {
    meson setup --help 2>/dev/null | grep -q -- '--prefer-static'
}

build_libplacebo() {
    local meson_args=(
        --buildtype=release
        --default-library=static
        -Dvulkan=enabled
        -Dvulkan-registry="$VULKAN_SDK/share/vulkan/registry/vk.xml"
        -Dopengl=disabled
        -Dd3d11=disabled
        -Ddemos=false
        --prefix "$BUILD_DIR"
        --libdir "$BUILD_DIR/lib"
    )

    if meson_supports_prefer_static; then
        meson_args+=(--prefer-static)
    fi

    if [ -f build/build.ninja ]; then
        meson setup --reconfigure "${meson_args[@]}" build || \
        meson setup --wipe "${meson_args[@]}" build
    else
        meson setup "${meson_args[@]}" build
    fi
}

# libplacebo (for GPU tone mapping)
if [ "$BUILD_LIBPLACEBO" = "1" ]; then
    # Download Vulkan SDK tarball (apt packages deprecated May 2025)
    VULKAN_SDK_VERSION=${VULKAN_SDK_VERSION:-1.4.335.0}
    VULKAN_SDK_DIR="${SRC_DIR}/vulkan-sdk-${VULKAN_SDK_VERSION}"
    if [ ! -d "$VULKAN_SDK_DIR" ]; then
        echo "Downloading Vulkan SDK $VULKAN_SDK_VERSION..."
        cd "$SRC_DIR"
        rm -f vulkansdk.tar.xz    # Clean up any partial download
        wget --progress=dot:giga -O vulkansdk.tar.xz "https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz"
        tar xf vulkansdk.tar.xz
        mv "${VULKAN_SDK_VERSION}" "vulkan-sdk-${VULKAN_SDK_VERSION}"
        rm -f vulkansdk.tar.xz
    fi
    export VULKAN_SDK="$VULKAN_SDK_DIR/x86_64"
    export PATH="$VULKAN_SDK/bin:$PATH"
    export PKG_CONFIG_PATH="$VULKAN_SDK/lib/pkgconfig:$PKG_CONFIG_PATH"
    echo "Using Vulkan SDK: $VULKAN_SDK"

    # Use static shaderc (avoid runtime .so dependency)
    if [ ! -f "$VULKAN_SDK/lib/pkgconfig/shaderc.pc.bak" ]; then
        cp "$VULKAN_SDK/lib/pkgconfig/shaderc.pc" "$VULKAN_SDK/lib/pkgconfig/shaderc.pc.bak"
    fi
    cp "$VULKAN_SDK/lib/pkgconfig/shaderc_combined.pc" "$VULKAN_SDK/lib/pkgconfig/shaderc.pc"

    cd "$SRC_DIR"
    git_update_or_clone "https://code.videolan.org/videolan/libplacebo.git" "libplacebo" 1
    cd libplacebo
    if [ -n "$LIBPLACEBO_GIT_REF" ]; then
        git fetch --depth 1 origin "$LIBPLACEBO_GIT_REF"
        git checkout -q FETCH_HEAD
    fi
    KEY=$(step_key "$SRC_DIR/libplacebo" "$VULKAN_SDK_VERSION")
    if ! step_done libplacebo "$KEY" "$BUILD_DIR/lib/libplacebo.a"; then
        build_libplacebo
        ninja -C build
        ninja -C build install
        step_mark libplacebo "$KEY"
    fi
fi

# LibTorch (PyTorch C++ library for DNN backend)
# Enables AI-based video filters like dnn_processing for upscaling, denoising, etc.
# NOTE: LibTorch 2.6.0+ renamed initXPU() to init(). We patch ffmpeg to handle both.
#       Use 2.7.0+ for RTX 50-series (Blackwell/SM 12.0) support.
LIBTORCH_FLAGS=()
if [ "$ENABLE_LIBTORCH" = "1" ]; then
    LIBTORCH_VERSION=${LIBTORCH_VERSION:-2.7.0}
    LIBTORCH_DIR="$SRC_DIR/libtorch"

    # Determine LibTorch CUDA variant
    # LIBTORCH_VARIANT: cu124 (default), auto, cpu, cu126, cu128, cu130, rocm6.4
    # PyTorch only releases for even-numbered CUDA versions
    if [ "$LIBTORCH_VARIANT" != "auto" ]; then
        # Validate user-provided variant
        if ! validate_torch_variant "$LIBTORCH_VARIANT"; then
            log_error "Valid variants: cu124, cu126, cu128, cu130, rocm6.4, cpu"
            exit 1
        fi
        TORCH_VARIANT="$LIBTORCH_VARIANT"
        echo "LibTorch: using $TORCH_VARIANT (explicit)"
    elif [ "$ENABLE_NVIDIA_CUDA" = "1" ]; then
        CUDA_MAJOR="${CUDA_VERSION%%-*}"
        CUDA_MINOR="${CUDA_VERSION#*-}"
        if [ "$CUDA_MAJOR" -ge 13 ]; then
            TORCH_VARIANT="cu130"
        else
            # Round down to nearest even, clamp to [6, 8]
            EVEN_MINOR=$(( (CUDA_MINOR / 2) * 2 ))
            [ "$EVEN_MINOR" -gt 8 ] && EVEN_MINOR=8
            [ "$EVEN_MINOR" -lt 6 ] && EVEN_MINOR=6
            TORCH_VARIANT="cu12${EVEN_MINOR}"
        fi
        echo "LibTorch: using $TORCH_VARIANT (from CUDA $CUDA_VERSION)"
    else
        TORCH_VARIANT="cpu"
        echo "LibTorch: using CPU-only variant"
    fi

    # Download LibTorch if not present or wrong variant
    LIBTORCH_MARKER="$LIBTORCH_DIR/.variant-${TORCH_VARIANT}"
    if [ ! -f "$LIBTORCH_MARKER" ]; then
        echo "Downloading LibTorch $LIBTORCH_VERSION ($TORCH_VARIANT)..."
        cd "$SRC_DIR"
        # Clean up any existing libtorch directory (safe: inside $SRC_DIR)
        safe_rm_rf "$SRC_DIR/libtorch" 3
        rm -f libtorch.zip

        # Download from pytorch.org
        # cu124 and earlier use cxx11-abi prefix, cu130+ dropped it
        if [[ "$TORCH_VARIANT" == cu13* ]] || [[ "$TORCH_VARIANT" == cu14* ]]; then
            LIBTORCH_URL="https://download.pytorch.org/libtorch/${TORCH_VARIANT}/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2B${TORCH_VARIANT}.zip"
        else
            LIBTORCH_URL="https://download.pytorch.org/libtorch/${TORCH_VARIANT}/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2B${TORCH_VARIANT}.zip"
        fi
        if ! wget --progress=dot:giga -O libtorch.zip "$LIBTORCH_URL"; then
            log_error "Failed to download LibTorch from $LIBTORCH_URL"
            exit 1
        fi
        unzip -q libtorch.zip
        rm -f libtorch.zip
        # Verify extraction succeeded before marking complete
        if [ ! -f "$LIBTORCH_DIR/lib/libtorch.so" ]; then
            log_error "LibTorch extraction failed - libtorch.so not found"
            exit 1
        fi
        touch "$LIBTORCH_MARKER"
    fi

    export LIBTORCH_PATH="$LIBTORCH_DIR"
    LIBTORCH_FLAGS=(--enable-libtorch)
    echo "Using LibTorch: $LIBTORCH_PATH"

    # Copy libtorch shared libs to permanent location (LIB_DIR)
    echo "Installing libtorch libs to $LIB_DIR..."
    if ! cp -a "$LIBTORCH_DIR/lib"/*.so* "$LIB_DIR/" 2>/dev/null; then
        log_warn "Some libtorch libs may not have been copied - check $LIB_DIR"
    fi
    # Verify at least libtorch.so was copied
    if [ ! -f "$LIB_DIR/libtorch.so" ]; then
        log_error "libtorch.so not found in $LIB_DIR after copy"
        exit 1
    fi

    # Create pkg-config file for libtorch (FFmpeg configure uses pkg-config for detection)
    mkdir -p "$BUILD_DIR/lib/pkgconfig"
    # Include CUDA libs if using CUDA variant
    if [[ "$TORCH_VARIANT" == cu* ]]; then
        TORCH_LIBS="-ltorch -lc10 -ltorch_cpu -ltorch_cuda -lc10_cuda"
        # Needed for ffmpeg extra-libs to ensure libtorch_cuda is linked (not just dlopen'd)
        TORCH_EXTRA_LIBS="-lc10_cuda -ltorch_cuda"
    else
        TORCH_LIBS="-ltorch -lc10 -ltorch_cpu"
        TORCH_EXTRA_LIBS=""
    fi
    cat > "$BUILD_DIR/lib/pkgconfig/libtorch.pc" << PCEOF
prefix=$LIBTORCH_DIR
exec_prefix=\${prefix}
libdir=$LIB_DIR
includedir=\${prefix}/include

Name: libtorch
Description: PyTorch C++ library
Version: $LIBTORCH_VERSION
Libs: -L\${libdir} $TORCH_LIBS
Cflags: -I\${includedir} -I\${includedir}/torch/csrc/api/include -std=c++17
PCEOF
    echo "Created libtorch.pc for pkg-config detection (variant: $TORCH_VARIANT)"
fi

# ffmpeg
FFMPEG_DIR="ffmpeg-${FFMPEG_VERSION}"
cd "$SRC_DIR"
if [ ! -d "$FFMPEG_DIR" ]; then
    if [ "$FFMPEG_VERSION" = "snapshot" ]; then
        rm -f ffmpeg-snapshot.tar.bz2    # Clean up any partial download
        wget --progress=dot:giga -O ffmpeg-snapshot.tar.bz2 https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2
        tar xjf ffmpeg-snapshot.tar.bz2
        mv ffmpeg "$FFMPEG_DIR"
        rm -f ffmpeg-snapshot.tar.bz2
    else
        rm -f "ffmpeg-${FFMPEG_VERSION}.tar.xz"    # Clean up any partial download
        wget --progress=dot:giga -O "ffmpeg-${FFMPEG_VERSION}.tar.xz" "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz"
        tar xJf "ffmpeg-${FFMPEG_VERSION}.tar.xz"
        rm -f "ffmpeg-${FFMPEG_VERSION}.tar.xz"
    fi
fi

# libstdc++ >= 14 headers fail under nvcc -std=c++11 (variable templates); ffmpeg's configure hardcodes it
if grep -q 'nvccflags="$nvccflags -std=c++11"' "$FFMPEG_DIR/configure"; then
    sed -i 's/nvccflags="$nvccflags -std=c++11"/nvccflags="$nvccflags -std=c++14"/' "$FFMPEG_DIR/configure"
fi

# Patch ffmpeg for SVT-AV1 v4.0.0 API compatibility
# v4.0.0 renamed enable_adaptive_quantization -> aq_mode
SVTAV1_CODEC="$FFMPEG_DIR/libavcodec/libsvtav1.c"
SVTAV1_HEADER="$BUILD_DIR/include/svt-av1/EbSvtAv1Enc.h"
if [ -f "$SVTAV1_CODEC" ] && grep -q "enable_adaptive_quantization" "$SVTAV1_CODEC" && \
   [ -f "$SVTAV1_HEADER" ] && grep -q "uint8_t aq_mode;" "$SVTAV1_HEADER"; then
    echo "Patching ffmpeg for SVT-AV1 v4.0.0 (adding compat alias)..."
    sed -i '/#include <EbSvtAv1ErrorCodes.h>/i\
/* SVT-AV1 v4.0.0 compat: renamed enable_adaptive_quantization -> aq_mode */\
#define enable_adaptive_quantization aq_mode' "$SVTAV1_CODEC"
fi

# Patch ffmpeg's torch backend
if [ "$ENABLE_LIBTORCH" = "1" ]; then
    TORCH_BACKEND="$FFMPEG_DIR/libavfilter/dnn/dnn_backend_torch.cpp"

    # Patch 1: Fix initXPU() -> init() for libtorch 2.6+ compatibility
    if [ -f "$TORCH_BACKEND" ] && grep -q "initXPU()" "$TORCH_BACKEND"; then
        TORCH_MAJOR=$(echo "$LIBTORCH_VERSION" | cut -d. -f1)
        TORCH_MINOR=$(echo "$LIBTORCH_VERSION" | cut -d. -f2)
        if [ "$TORCH_MAJOR" -gt 2 ] || { [ "$TORCH_MAJOR" -eq 2 ] && [ "$TORCH_MINOR" -ge 6 ]; }; then
            echo "Patching ffmpeg for libtorch 2.6+ (initXPU -> init)..."
            sed -i 's/initXPU()/init()/g' "$TORCH_BACKEND"
        fi
    fi

    # Patch 2: CUDA device support. ffmpeg links only libtorch_cpu, so torch's CUDA
    # hooks register only once libtorch_cuda.so is loaded; any is_available() check
    # before that load reports no CUDA. Newer upstream ships its own CUDA branch
    # (with exactly that early check), so ensure the branch exists, then load
    # libtorch_cuda.so as the branch's first statement.
    if [ -f "$TORCH_BACKEND" ] && ! grep -q "device.is_cuda()" "$TORCH_BACKEND"; then
        echo "Patching ffmpeg torch backend for CUDA device support..."
        sed -i '/at::detail::getXPUHooks().init/a\
    } else if (device.is_cuda()) {\
        if (!at::cuda::is_available()) {\
            av_log(ctx, AV_LOG_ERROR, "No CUDA device found\\n");\
            goto fail;\
        }' "$TORCH_BACKEND"
        if ! grep -q "#include <ATen/cuda/CUDAContext.h>" "$TORCH_BACKEND"; then
            sed -i '/#include <torch\/torch.h>/a #include <ATen/cuda/CUDAContext.h>' "$TORCH_BACKEND"
        fi
    fi
    if [ -f "$TORCH_BACKEND" ] && ! grep -q 'dlopen("libtorch_cuda.so"' "$TORCH_BACKEND"; then
        echo "Patching ffmpeg torch backend to load libtorch_cuda.so..."
        if ! grep -q "#include <dlfcn.h>" "$TORCH_BACKEND"; then
            sed -i '/#include <torch\/script.h>/a #include <dlfcn.h>' "$TORCH_BACKEND"
        fi
        sed -i '/} else if (device.is_cuda()) {/a\
        static bool cuda_lib_loaded = false;\
        if (!cuda_lib_loaded) {\
            cuda_lib_loaded = true;\
            if (!dlopen("libtorch_cuda.so", RTLD_NOW | RTLD_GLOBAL))\
                av_log(ctx, AV_LOG_WARNING, "Failed to load libtorch_cuda.so: %s\\n", dlerror());\
        }' "$TORCH_BACKEND"
        verify_patch "$TORCH_BACKEND" 'dlopen("libtorch_cuda.so"' "torch backend loads libtorch_cuda.so" || exit 1
        echo "Torch CUDA patch applied"
    fi

    # Patch 3: Add TensorRT support (load runtime + handle tuple outputs)
    if [ -f "$TORCH_BACKEND" ] && ! grep -q "isTuple" "$TORCH_BACKEND"; then
        echo "Patching ffmpeg torch backend for TensorRT support..."

        # Add dlfcn.h header for dlopen
        if ! grep -q "#include <dlfcn.h>" "$TORCH_BACKEND"; then
            sed -i '/#include <torch\/script.h>/a #include <dlfcn.h>' "$TORCH_BACKEND"
        fi

        # Add TensorRT runtime loading in model init (before torch::jit::load)
        if ! grep -q "libtorchtrt_runtime" "$TORCH_BACKEND"; then
            sed -i '/torch::jit::load(ctx->model_filename)/i\
    // Load TensorRT runtime if available (enables TRT-compiled models)\
    static bool trt_init_attempted = false;\
    if (!trt_init_attempted) {\
        trt_init_attempted = true;\
        void *trt_handle = dlopen("libtorchtrt_runtime.so", RTLD_NOW | RTLD_GLOBAL);\
        if (trt_handle) {\
            av_log(ctx, AV_LOG_INFO, "TensorRT runtime loaded\\n");\
        }\
    }' "$TORCH_BACKEND"
        fi

        # Change forward().toTensor() to handle TRT tuple outputs
        sed -i 's/\*infer_request->output = th_model->jit_model->forward(inputs)\.toTensor();/auto _fwd_out = th_model->jit_model->forward(inputs);\
    if (_fwd_out.isTuple()) {\
        *infer_request->output = _fwd_out.toTuple()->elements()[0].toTensor();\
    } else {\
        *infer_request->output = _fwd_out.toTensor();\
    }/' "$TORCH_BACKEND"

        # Fix device detection for TRT models (they may not have parameters)
        sed -i 's/c10::Device device = (\*th_model->jit_model->parameters()\.begin())\.device();/c10::Device device = torch::kCUDA;\
    auto params = th_model->jit_model->parameters();\
    if (params.begin() != params.end()) {\
        device = (*params.begin()).device();\
    }/' "$TORCH_BACKEND"

        echo "Torch TensorRT patch applied"
    fi
fi

# Patch ffmpeg's libplacebo filter to include libavformat version header
# (suppresses LIBAVFORMAT_VERSION_INT -Wundef warnings).
LIBPLACEBO_FILTER="$FFMPEG_DIR/libavfilter/vf_libplacebo.c"
if [ -f "$LIBPLACEBO_FILTER" ] && ! grep -q "libavformat/version.h" "$LIBPLACEBO_FILTER"; then
    sed -i '/#include "libavfilter\/avfilter.h"/a #include "libavformat/version.h"' "$LIBPLACEBO_FILTER"
fi

# TensorRT native backend (no libtorch dependency)
# Loads pre-compiled .engine files directly for maximum performance
TENSORRT_FLAGS=()
if [ "$ENABLE_TENSORRT" = "1" ]; then
    echo "Patching FFmpeg for native TensorRT DNN backend..."
    PATCH_DIR="$SCRIPT_DIR/patches"

    # Copy TensorRT backend source file
    if [ -f "$PATCH_DIR/dnn_backend_tensorrt.cpp" ]; then
        cp "$PATCH_DIR/dnn_backend_tensorrt.cpp" "$FFMPEG_DIR/libavfilter/dnn/"
        echo "Copied dnn_backend_tensorrt.cpp"
    else
        echo "WARNING: dnn_backend_tensorrt.cpp not found in $PATCH_DIR"
    fi

    # Copy CUDA kernels for GPU-resident format conversion (zero-copy)
    # Rename to .cuda to prevent FFmpeg from auto-compiling it as .cu -> .o
    # Our custom PTX rules in the Makefile will compile it properly
    if [ -f "$PATCH_DIR/dnn_cuda_kernels.cu" ]; then
        cp "$PATCH_DIR/dnn_cuda_kernels.cu" "$FFMPEG_DIR/libavfilter/dnn/dnn_cuda_kernels.cuda"
        cp "$PATCH_DIR/dnn_cuda_kernels.h" "$FFMPEG_DIR/libavfilter/dnn/"
        echo "Copied CUDA format conversion kernels (renamed to .cuda to avoid auto-build)"
    else
        echo "WARNING: dnn_cuda_kernels.cu not found in $PATCH_DIR"
    fi

    # Copy patched vf_dnn_processing.c for CUDA frame support
    if [ -f "$PATCH_DIR/vf_dnn_processing.c" ]; then
        cp "$PATCH_DIR/vf_dnn_processing.c" "$FFMPEG_DIR/libavfilter/"
        echo "Copied vf_dnn_processing.c with CUDA frame support"
    fi

    # Patch dnn_interface.h to add DNN_TRT enum and TRTOptions
    DNN_INTERFACE_H="$FFMPEG_DIR/libavfilter/dnn_interface.h"
    if [ -f "$DNN_INTERFACE_H" ] && ! grep -q "DNN_TRT" "$DNN_INTERFACE_H"; then
        # FFmpeg 7.0 has single-line enum: {DNN_TF = 1, DNN_OV, DNN_TH}
        # FFmpeg 7.1+ has multi-line bit flags (DNN_TH = 1 << 2, later DNN_ONNX = 1 << 3, ...)
        if grep -qE "^ *DNN_TH = 1 << [0-9]+" "$DNN_INTERFACE_H"; then
            # Multi-line bit flags: append DNN_TRT as the next free bit after the last member
            awk '
                /^typedef enum \{/ { in_enum = 1 }
                in_enum && /^ *DNN_[A-Z0-9_]+ = 1 << [0-9]+/ {
                    n = $0; sub(/.*<< */, "", n); sub(/[^0-9].*/, "", n)
                    if (n + 0 > max) max = n + 0
                    last = NR
                }
                in_enum && /^\} DNNBackendType;/ { close_line = NR; in_enum = 0 }
                { lines[NR] = $0 }
                END {
                    for (i = 1; i <= NR; i++) {
                        if (i == last && lines[i] !~ /,[[:space:]]*$/) lines[i] = lines[i] ","
                        if (i == close_line) printf "    DNN_TRT = 1 << %d\n", max + 1
                        print lines[i]
                    }
                }' "$DNN_INTERFACE_H" > "$DNN_INTERFACE_H.tmp" && mv "$DNN_INTERFACE_H.tmp" "$DNN_INTERFACE_H"
        elif grep -q "DNN_TF = 1, DNN_OV, DNN_TH}" "$DNN_INTERFACE_H"; then
            # Single-line format (7.0)
            sed -i 's/DNN_TF = 1, DNN_OV, DNN_TH}/DNN_TF = 1, DNN_OV, DNN_TH, DNN_TRT}/' "$DNN_INTERFACE_H"
        else
            echo "ERROR: Unknown DNNBackendType enum format in dnn_interface.h" >&2
            grep -A5 "DNNBackendType" "$DNN_INTERFACE_H" >&2
            exit 1
        fi
        # Add TRTOptions struct after THOptions
        sed -i '/^} THOptions;$/a\
\
typedef struct TRTOptions {\
    const AVClass *clazz;\
    int device_id;\
} TRTOptions;' "$DNN_INTERFACE_H"
        # Add trt_option to DnnContext (after torch_option)
        sed -i '/#if CONFIG_LIBTORCH/,/#endif/{
            /#endif/a\
#if CONFIG_LIBTENSORRT\
    TRTOptions trt_option;\
#endif
        }' "$DNN_INTERFACE_H"
        # Verify patch was applied
        if grep -q "DNN_TRT" "$DNN_INTERFACE_H"; then
            echo "Patched dnn_interface.h"
        else
            echo "ERROR: Failed to patch dnn_interface.h - DNN_TRT not found after patching" >&2
            exit 1
        fi
    fi

    # Patch dnn_interface.c to register TensorRT backend
    DNN_INTERFACE_C="$FFMPEG_DIR/libavfilter/dnn/dnn_interface.c"
    if [ -f "$DNN_INTERFACE_C" ] && ! grep -q "ff_dnn_backend_tensorrt" "$DNN_INTERFACE_C"; then
        # Add extern declaration
        sed -i '/extern const DNNModule ff_dnn_backend_torch;/a extern const DNNModule ff_dnn_backend_tensorrt;' "$DNN_INTERFACE_C"
        # Add to backend list
        sed -i '/#if CONFIG_LIBTORCH/,/#endif/{
            /#endif/a\
#if CONFIG_LIBTENSORRT\
        {offsetof(DnnContext, trt_option), .module = \&ff_dnn_backend_tensorrt},\
#endif
        }' "$DNN_INTERFACE_C"
        # Verify patch was applied
        if ! verify_patch "$DNN_INTERFACE_C" "ff_dnn_backend_tensorrt" "dnn_interface.c TensorRT registration"; then
            exit 1
        fi
        log_info "Patched dnn_interface.c"
    fi

    # Patch dnn/Makefile to add TensorRT objects and CUDA kernel PTX compilation
    DNN_MAKEFILE="$FFMPEG_DIR/libavfilter/dnn/Makefile"
    if [ -f "$DNN_MAKEFILE" ] && ! grep -q "CONFIG_LIBTENSORRT" "$DNN_MAKEFILE"; then
        # Add TensorRT backend object
        sed -i '/CONFIG_LIBTORCH.*dnn_backend_torch/a DNN-OBJS-$(CONFIG_LIBTENSORRT)             += dnn/dnn_backend_tensorrt.o' "$DNN_MAKEFILE"
        # Add embedded PTX object (compiled PTX as C byte array)
        sed -i '/dnn_backend_tensorrt.o/a DNN-OBJS-$(CONFIG_LIBTENSORRT)               += dnn/dnn_cuda_kernels_ptx.o' "$DNN_MAKEFILE"
        # Add PTX compilation and embedding rules
        # 1. Compile .cu to .ptx with nvcc
        # 2. Embed .ptx as C byte array using xxd (bin2c alternative)
        # 3. Compile embedded C to object
        cat >> "$DNN_MAKEFILE" << 'PTXRULES'

# CUDA kernel PTX compilation and embedding (no cudart dependency)
# Step 1: Compile CUDA kernels to PTX (intermediate representation)
# Source is .cuda (not .cu) to prevent FFmpeg from auto-compiling to .o
libavfilter/dnn/dnn_cuda_kernels.ptx: libavfilter/dnn/dnn_cuda_kernels.cuda
	$(NVCC) --ptx -o $@ -x cu $< -m64

# Step 2: Embed PTX as C byte array (using xxd, like bin2c)
libavfilter/dnn/dnn_cuda_kernels_ptx.c: libavfilter/dnn/dnn_cuda_kernels.ptx
	@echo "Embedding PTX as C byte array..."
	@echo "/* Auto-generated - do not edit */" > $@
	@echo "#include <stddef.h>" >> $@
	@echo "" >> $@
	@printf "const unsigned char ff_dnn_cuda_kernels_ptx[] = {\n" >> $@
	@xxd -i < $< >> $@
	@echo "};" >> $@
	@echo "" >> $@
	@printf "const unsigned int ff_dnn_cuda_kernels_ptx_len = sizeof(ff_dnn_cuda_kernels_ptx);\n" >> $@

# Step 3: Compile embedded PTX C file to object
libavfilter/dnn/dnn_cuda_kernels_ptx.o: libavfilter/dnn/dnn_cuda_kernels_ptx.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<
PTXRULES
        # Verify Makefile patch
        if ! verify_patch "$DNN_MAKEFILE" "dnn_cuda_kernels_ptx.o" "dnn/Makefile PTX rules"; then
            exit 1
        fi
        log_info "Patched dnn/Makefile with TensorRT and CUDA PTX kernel support"
    fi

    # Patch configure to add --enable-libtensorrt option
    CONFIGURE="$FFMPEG_DIR/configure"
    if [ -f "$CONFIGURE" ] && ! grep -q "enable-libtensorrt" "$CONFIGURE"; then
        # Add help text
        sed -i '/--enable-libtorch.*enable Torch/a\  --enable-libtensorrt     enable TensorRT as one DNN backend [no]' "$CONFIGURE"
        # Add to library list
        sed -i '/^    libtorch$/a\    libtensorrt' "$CONFIGURE"
        # Add to dnn_deps_any
        sed -i 's/dnn_deps_any="libtensorflow libopenvino libtorch"/dnn_deps_any="libtensorflow libopenvino libtorch libtensorrt"/' "$CONFIGURE"
        # Add header check (after libtorch check)
        # TensorRT (libnvinfer) and CUDA (libcuda) are loaded via dlopen at runtime.
        # CUDA kernels are compiled to PTX and loaded via Driver API - no cudart dependency.
        sed -i '/enabled libtorch.*require_cxx libtorch/a\
enabled libtensorrt       && check_cxxflags -std=c++17 && check_headers NvInfer.h' "$CONFIGURE"
        echo "Patched configure (TensorRT via dlopen, CUDA linked for kernels)"
    fi

    TENSORRT_FLAGS=(--enable-libtensorrt)
    echo "TensorRT DNN backend patches applied"
fi

cd "$FFMPEG_DIR"
# Build configure flags
# MARCH=native for CPU-specific optimizations (opt-in, not portable)
EXTRA_CFLAGS="-I$BUILD_DIR/include -O3${MARCH:+ -march=$MARCH -mtune=$MARCH}"
EXTRA_CXXFLAGS=""
# -rpath embeds library search path in binary so it finds our built libs at runtime
EXTRA_LDFLAGS="-L$BUILD_DIR/lib -s -Wl,-rpath,$LIB_DIR"
if [ "$ENABLE_NVIDIA_CUDA" = "1" ]; then
    EXTRA_CFLAGS="$EXTRA_CFLAGS -I$CUDA_PATH/include"
    EXTRA_LDFLAGS="$EXTRA_LDFLAGS -L$CUDA_PATH/lib64"
fi
if [ "$BUILD_LIBPLACEBO" = "1" ]; then
    EXTRA_CFLAGS="$EXTRA_CFLAGS -I$VULKAN_SDK/include"
    EXTRA_LDFLAGS="$EXTRA_LDFLAGS -L$VULKAN_SDK/lib"
fi
if [ "$ENABLE_LIBTORCH" = "1" ]; then
    # LibTorch needs C++ flags (FFmpeg uses require_cxx for libtorch detection)
    # Include CUDA path for CUDA torch support
    EXTRA_CXXFLAGS="-I$LIBTORCH_PATH/include -I$LIBTORCH_PATH/include/torch/csrc/api/include"
    if [ "$ENABLE_NVIDIA_CUDA" = "1" ]; then
        EXTRA_CXXFLAGS="$EXTRA_CXXFLAGS -I$CUDA_PATH/include"
    fi
    EXTRA_LDFLAGS="$EXTRA_LDFLAGS -L$LIB_DIR -Wl,-rpath,$LIB_DIR"
fi
if [ "$ENABLE_TENSORRT" = "1" ]; then
    # TensorRT needs C++ flags with CUDA headers (uses require_cxx for detection)
    # Note: We use CUDA Driver API (libcuda.so) loaded via dlopen at runtime,
    # NOT CUDA Runtime API (libcudart.so). This avoids load-time dependency.
    if [ "$ENABLE_NVIDIA_CUDA" = "1" ]; then
        EXTRA_CXXFLAGS="$EXTRA_CXXFLAGS -I$CUDA_PATH/include"
    fi
fi
CONFIGURE_CMD=(
    ./configure
    --prefix="$BUILD_DIR"
    --pkg-config-flags="--static"
    --extra-cflags="$EXTRA_CFLAGS"
    --extra-cxxflags="$EXTRA_CXXFLAGS"
    --extra-ldflags="$EXTRA_LDFLAGS"
    --extra-libs="-lpthread -lm -ldl${TORCH_EXTRA_LIBS:+ $TORCH_EXTRA_LIBS}"
    --ld="g++"
    --bindir="$BIN_DIR"
    --disable-debug
    --enable-gpl
    --enable-version3
    --enable-openssl
    --enable-libaom
    --enable-libass
    --enable-libbluray
    --enable-libfdk-aac
    --enable-libfontconfig
    --enable-libfreetype
    --enable-libfribidi
    --enable-libharfbuzz
    --enable-libjxl
    --enable-libmp3lame
    --enable-libopus
    --enable-libsvtav1
    --enable-libdav1d
    --enable-libvmaf
    --enable-libvorbis
    --enable-libvpx
    --enable-libwebp
    --enable-libx264
    --enable-libx265
    --enable-librubberband
    --enable-libsoxr
    --enable-libsrt
    --enable-libvidstab
    --enable-libvpl
    --enable-libzimg
    --enable-opencl
    --enable-vaapi
    --enable-nonfree
    "${CUDA_FLAGS[@]}"
    "${AMF_FLAGS[@]}"
    "${LIBTORCH_FLAGS[@]}"
    "${TENSORRT_FLAGS[@]}"
)

if [ "$BUILD_LIBPLACEBO" = "1" ]; then
    CONFIGURE_CMD+=(--enable-vulkan --enable-libplacebo)
fi

if [ -n "$NVCC_ARCH" ]; then
    CONFIGURE_CMD+=(--nvccflags="$NVCC_ARCH")
fi

# Build PATH: include CUDA bin if NVIDIA enabled
BUILD_PATH="$BIN_DIR:$PATH"
[ "$ENABLE_NVIDIA_CUDA" = "1" ] && BUILD_PATH="$CUDA_PATH/bin:$BUILD_PATH"

# ffmpeg key: everything it links or is patched with. In-place source patches above are
# idempotent, so the script and patches/ hashes stand in for "what patches were applied".
FFMPEG_KEY=$(step_key "$SRC_DIR/$FFMPEG_DIR" \
    "$(cat VERSION RELEASE 2>/dev/null)" \
    "${CONFIGURE_CMD[*]}" \
    "$(cd "$STAMP_DIR" 2>/dev/null && for f in $(ls | sort); do [ "$f" = ffmpeg ] || echo "$f=$(cat "$f")"; done)" \
    "$( [ "$ENABLE_NVIDIA_CUDA" = "1" ] && "$CUDA_PATH/bin/nvcc" --version 2>/dev/null | tail -1)" \
    "${LIBTORCH_VERSION:-}:${TORCH_VARIANT:-}" \
    "${NVCC_CCBIN:-}" \
    "$(git -C "$SRC_DIR/nv-codec-headers" rev-parse HEAD 2>/dev/null)" \
    "$(sha256sum "$SCRIPT_DIR/install-ffmpeg.sh" "$SCRIPT_DIR"/patches/* 2>/dev/null | cut -c1-64)")
if ! step_done ffmpeg "$FFMPEG_KEY" "$BIN_DIR/ffmpeg"; then
    PATH="$BUILD_PATH" PKG_CONFIG_PATH="$BUILD_DIR/lib/pkgconfig:$PKG_CONFIG_PATH" "${CONFIGURE_CMD[@]}"
    PATH="$BUILD_PATH" make -j "$NPROC"
    make install
    step_mark ffmpeg "$FFMPEG_KEY"
fi
hash -r

grep -q "$BUILD_DIR/share/man" "$HOME/.manpath" 2>/dev/null || echo "MANPATH_MAP $BIN_DIR $BUILD_DIR/share/man" >> "$HOME/.manpath"

log_info "FFmpeg build completed successfully"
