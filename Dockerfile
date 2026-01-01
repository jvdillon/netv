# Multi-stage build: compile FFmpeg with all HW accel, then copy to slim runtime
#
# We use Ubuntu 24.04 for BOTH stages because:
# - Newer codec libraries (libvpx 1.14, x265, etc.) = better quality/performance
# - libfdk-aac available without non-free repo config
# - Better hardware encoding support (NVENC, VAAPI, QSV headers)
# - Matching library ABIs between builder and runtime (critical!)
#
# Using mismatched distros (e.g., Ubuntu builder + Debian runtime) breaks at
# runtime due to shared library soname mismatches (libvpx.so.9 vs .so.7, etc.)

# =============================================================================
# Stage 1: Build FFmpeg with NVENC, VAAPI, QSV + all codecs
# =============================================================================
FROM ubuntu:24.04 AS builder

# Override for older GPUs (e.g., --build-arg CUDA_VERSION=12-9 for Maxwell/Pascal)
ARG CUDA_VERSION=""

ENV DEBIAN_FRONTEND=noninteractive
ENV FFMPEG_BUILD=/opt/ffmpeg_build
ENV PATH="/opt/bin:/usr/local/cuda/bin:$PATH"

# Add CUDA repo and install build dependencies
RUN apt-get update && apt-get install -y wget gnupg && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y \
    autoconf \
    automake \
    build-essential \
    cmake \
    git \
    meson \
    nasm \
    ninja-build \
    pkg-config \
    texinfo \
    yasm \
    libaom-dev \
    libass-dev \
    libdav1d-dev \
    libfdk-aac-dev \
    libffmpeg-nvenc-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libsoxr-dev \
    libsrt-openssl-dev \
    libssl-dev \
    libunistring-dev \
    libwebp-dev \
    libzimg-dev \
    liblzma-dev \
    liblzo2-dev \
    libmp3lame-dev \
    libnuma-dev \
    libopus-dev \
    libsdl2-dev \
    libtool \
    libva-dev \
    libvdpau-dev \
    libvpl-dev \
    libvorbis-dev \
    libvpx-dev \
    libx264-dev \
    libx265-dev \
    libxcb-shm0-dev \
    libxcb-xfixes0-dev \
    libxcb1-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA (uses ARG if set, otherwise auto-detects latest)
RUN apt-get update && \
    if [ -z "$CUDA_VERSION" ]; then \
      CUDA_VERSION=$(apt-cache search '^cuda-nvcc-[0-9]' | sed 's/cuda-nvcc-//' | cut -d' ' -f1 | sort -V | tail -1); \
    fi && \
    echo "Installing CUDA version: $CUDA_VERSION" && \
    apt-get install -y cuda-nvcc-$CUDA_VERSION cuda-cudart-dev-$CUDA_VERSION && \
    CUDA_DIR=$(ls -d /usr/local/cuda-* 2>/dev/null | head -1) && \
    echo "CUDA directory: $CUDA_DIR" && \
    ln -sf "$CUDA_DIR" /usr/local/cuda && \
    echo "nvcc location: $(which nvcc)" && \
    nvcc --version && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src

# nv-codec-headers (for NVENC - works at runtime if GPU present)
RUN git clone --depth 1 https://git.videolan.org/git/ffmpeg/nv-codec-headers.git && \
    cd nv-codec-headers && \
    make && \
    make PREFIX=$FFMPEG_BUILD install

# SVT-AV1
RUN git clone --depth 1 https://gitlab.com/AOMediaCodec/SVT-AV1.git && \
    mkdir -p SVT-AV1/build && \
    cd SVT-AV1/build && \
    cmake -G "Unix Makefiles" \
        -DCMAKE_INSTALL_PREFIX=$FFMPEG_BUILD \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_DEC=OFF \
        -DBUILD_SHARED_LIBS=OFF .. && \
    make -j$(nproc) && \
    make install

# libvmaf
RUN git clone --depth 1 https://github.com/Netflix/vmaf && \
    mkdir -p vmaf/libvmaf/build && \
    cd vmaf/libvmaf/build && \
    meson setup \
        -Denable_tests=false \
        -Denable_docs=false \
        --buildtype=release \
        --default-library=static \
        --prefix=$FFMPEG_BUILD \
        --libdir=$FFMPEG_BUILD/lib \
        .. && \
    ninja && \
    ninja install

# FFmpeg
RUN wget -O ffmpeg-snapshot.tar.bz2 https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2 && \
    tar xjf ffmpeg-snapshot.tar.bz2 && \
    cd ffmpeg && \
    PKG_CONFIG_PATH="$FFMPEG_BUILD/lib/pkgconfig" ./configure \
        --prefix=$FFMPEG_BUILD \
        --bindir=/opt/bin \
        --pkg-config-flags="--static" \
        --extra-cflags="-I$FFMPEG_BUILD/include -I/usr/local/cuda/include -O3 -march=native -mtune=native" \
        --extra-ldflags="-L$FFMPEG_BUILD/lib -L/usr/local/cuda/lib64 -s" \
        --extra-libs="-lpthread -lm" \
        --ld="g++" \
        --enable-gpl \
        --enable-version3 \
        --enable-openssl \
        --enable-libaom \
        --enable-libass \
        --enable-libfdk-aac \
        --enable-libfontconfig \
        --enable-libfreetype \
        --enable-libmp3lame \
        --enable-libopus \
        --enable-libsvtav1 \
        --enable-libdav1d \
        --enable-libvmaf \
        --enable-libvorbis \
        --enable-libvpx \
        --enable-libwebp \
        --enable-libx264 \
        --enable-libx265 \
        --enable-libzimg \
        --enable-libsoxr \
        --enable-libsrt \
        --enable-vaapi \
        --enable-libvpl \
        --enable-cuda-nvcc \
        --enable-nvenc \
        --enable-cuvid \
        --enable-nonfree && \
    make -j$(nproc) && \
    make install

# =============================================================================
# Stage 2: Runtime image
# =============================================================================
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Python + runtime libraries for FFmpeg + VAAPI/QSV drivers
# Use ldd /usr/local/bin/ffmpeg in builder to verify all deps are covered
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    # Core codec libs
    libaom3 \
    libass9 \
    libdav1d7 \
    libfdk-aac2 \
    libmp3lame0 \
    libopus0 \
    libvorbis0a \
    libvorbisenc2 \
    libvpx9 \
    libwebp7 \
    libwebpmux3 \
    libx264-164 \
    libx265-199 \
    # Text/font rendering
    libfontconfig1 \
    libfreetype6 \
    # Audio/video processing
    libsoxr0 \
    libzimg2 \
    libnuma1 \
    # Network/crypto
    libsrt1.5-openssl \
    libssl3 \
    # X11/display
    libxcb1 \
    libxcb-shm0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxv1 \
    libx11-6 \
    libxext6 \
    # Hardware accel
    libva2 \
    libva-drm2 \
    libva-x11-2 \
    libvdpau1 \
    libvpl2 \
    intel-media-va-driver \
    mesa-va-drivers \
    # Other deps
    zlib1g \
    libunistring5 \
    liblzma5 \
    liblzo2-2 \
    libasound2t64 \
    libdrm2 \
    libsndio7.0 \
    libsdl2-2.0-0 \
    libpulse0 \
    && rm -rf /var/lib/apt/lists/*

# Copy FFmpeg binaries from builder
COPY --from=builder /opt/bin/ffmpeg /usr/local/bin/
COPY --from=builder /opt/bin/ffprobe /usr/local/bin/

# Verify all shared libraries are available (fail build early if not)
RUN ldd /usr/local/bin/ffmpeg | grep -q "not found" && \
    { echo "Missing libraries:"; ldd /usr/local/bin/ffmpeg | grep "not found"; exit 1; } || \
    echo "All ffmpeg dependencies satisfied"

# App setup
WORKDIR /app
COPY pyproject.toml README.md ./
COPY *.py ./
COPY templates/ templates/
COPY static/ static/

# Install Python dependencies
RUN python3 -m pip install --no-cache-dir --break-system-packages .

# Runtime config
EXPOSE 8000
VOLUME /app/cache

ENV NETV_PORT=8000
ENV NETV_HTTPS=""
ENV LOG_LEVEL=INFO

# Run as non-root
RUN useradd -m netv
USER netv

# Shell form for env var expansion; NETV_HTTPS=1 adds --https flag
CMD python3 main.py --port ${NETV_PORT} ${NETV_HTTPS:+--https}
