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

ENV DEBIAN_FRONTEND=noninteractive
ENV FFMPEG_BUILD=/opt/ffmpeg_build
ENV PATH="/opt/bin:$PATH"

# Add CUDA repo for nvcc
RUN apt-get update && apt-get install -y wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update

# Build dependencies
RUN apt-get install -y \
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
    wget \
    yasm \
    libaom-dev \
    libass-dev \
    libdav1d-dev \
    libfdk-aac-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libsoxr-dev \
    libsrt-openssl-dev \
    libssl-dev \
    libwebp-dev \
    libzimg-dev \
    liblzma-dev \
    libmp3lame-dev \
    libnuma-dev \
    libopus-dev \
    libtool \
    libva-dev \
    libvdpau-dev \
    libvorbis-dev \
    libvpx-dev \
    libx264-dev \
    libx265-dev \
    libxcb-shm0-dev \
    libxcb-xfixes0-dev \
    libxcb1-dev \
    zlib1g-dev \
    # CUDA for nvcc (minimal, ~500MB vs full toolkit ~4GB)
    cuda-nvcc \
    && rm -rf /var/lib/apt/lists/*

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
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libass9 \
    libdav1d7 \
    libfdk-aac2 \
    libfontconfig1 \
    libfreetype6 \
    libsoxr0 \
    libsrt1.5-openssl \
    libssl3 \
    libwebp7 \
    libzimg2 \
    libmp3lame0 \
    libnuma1 \
    libopus0 \
    libva2 \
    libva-drm2 \
    libvdpau1 \
    libvorbis0a \
    libvorbisenc2 \
    libvpx9 \
    libx264-164 \
    libx265-199 \
    libxcb1 \
    # VAAPI drivers (Intel + AMD)
    intel-media-va-driver \
    mesa-va-drivers \
    && rm -rf /var/lib/apt/lists/*

# Copy FFmpeg binaries from builder
COPY --from=builder /opt/bin/ffmpeg /usr/local/bin/
COPY --from=builder /opt/bin/ffprobe /usr/local/bin/

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

# Run as non-root
RUN useradd -m netv
USER netv

# Shell form for env var expansion; NETV_HTTPS=1 adds --https flag
CMD python3 main.py --port ${NETV_PORT} ${NETV_HTTPS:+--https}
