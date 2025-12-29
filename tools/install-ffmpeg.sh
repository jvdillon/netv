#!/bin/bash
# Build ffmpeg from source with NVIDIA NVENC support
# https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu
set -e

# Build from source for latest versions and -march=native optimizations (set to 0 for apt packages)
BUILD_LIBAOM=${BUILD_LIBAOM:-0}
BUILD_NV_HEADERS=${BUILD_NV_HEADERS:-1}

# Build paths
SRC_DIR="${SRC_DIR:-$HOME/ffmpeg_sources}"
BUILD_DIR="${BUILD_DIR:-$HOME/ffmpeg_build}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"

NPROC=$(nproc)

# Add CUDA repo if not present
if ! dpkg -l cuda-keyring 2>/dev/null | grep -q ^ii; then
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  rm cuda-keyring_1.1-1_all.deb
  sudo apt-get update
fi

# Get latest CUDA version
CUDA_VERSION=$(apt-cache search '^cuda-nvcc-[0-9]' | sed 's/cuda-nvcc-//' | cut -d' ' -f1 | sort -V | tail -1)
echo "Using CUDA version: $CUDA_VERSION"

sudo apt install -y \
  autoconf \
  automake \
  build-essential \
  cmake \
  git-core \
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
  libffmpeg-nvenc-dev \
  libfontconfig1-dev \
  libfreetype6-dev \
  libsoxr-dev \
  libsrt-openssl-dev \
  libssl-dev \
  libwebp-dev \
  libzimg-dev \
  liblzma-dev \
  liblzo2-dev \
  libmp3lame-dev \
  libnuma-dev \
  libopus-dev \
  libsdl2-dev \
  libtool \
  libunistring-dev \
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
  cuda-nvcc-$CUDA_VERSION \
  cuda-cudart-dev-$CUDA_VERSION

mkdir -p "$SRC_DIR"

# libaom (optional - system package is usually sufficient)
if [ "$BUILD_LIBAOM" = "1" ]; then
  cd "$SRC_DIR" &&
  git -C aom pull 2> /dev/null || git clone --depth 1 https://aomedia.googlesource.com/aom &&
  mkdir -p aom_build &&
  cd aom_build &&
  PATH="$BIN_DIR:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" -DENABLE_TESTS=OFF -DENABLE_NASM=on ../aom &&
  PATH="$BIN_DIR:$PATH" make -j $NPROC &&
  make install
fi

# libsvtav1
cd "$SRC_DIR" && \
git -C SVT-AV1 pull 2> /dev/null || git clone --depth 1 https://gitlab.com/AOMediaCodec/SVT-AV1.git && \
mkdir -p SVT-AV1/build && \
cd SVT-AV1/build && \
PATH="$BIN_DIR:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DBUILD_DEC=OFF -DBUILD_SHARED_LIBS=OFF .. && \
PATH="$BIN_DIR:$PATH" make -j $NPROC && \
make install

# libvmaf
cd "$SRC_DIR" &&
git -C vmaf-master pull 2> /dev/null || git clone --depth 1 'https://github.com/Netflix/vmaf' 'vmaf-master' &&
mkdir -p 'vmaf-master/libvmaf/build' &&
cd 'vmaf-master/libvmaf/build' &&
if [ -f build.ninja ]; then
  meson setup --reconfigure -Denable_tests=false -Denable_docs=false --buildtype=release --default-library=static '../' --prefix "$BUILD_DIR" --bindir="$BIN_DIR" --libdir="$BUILD_DIR/lib"
else
  meson setup -Denable_tests=false -Denable_docs=false --buildtype=release --default-library=static '../' --prefix "$BUILD_DIR" --bindir="$BIN_DIR" --libdir="$BUILD_DIR/lib"
fi &&
ninja &&
ninja install


# nv-codec-headers (optional - system package is usually sufficient)
if [ "$BUILD_NV_HEADERS" = "1" ]; then
  cd "$SRC_DIR" &&
  git -C nv-codec-headers pull 2> /dev/null || git clone --depth 1 https://git.videolan.org/git/ffmpeg/nv-codec-headers.git &&
  cd nv-codec-headers &&
  make &&
  make PREFIX="$BUILD_DIR" install
fi

# CUDA flags (always enabled since we install CUDA packages)
CUDA_FLAGS="--enable-cuda-nvcc --enable-nvenc --enable-cuvid"
NVCC_GENCODE=""

# Detect GPU compute capability for optimized nvcc flags
if command -v nvidia-smi &> /dev/null; then
  COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  if [ -n "$COMPUTE_CAP" ]; then
    COMPUTE_CAP_NUM=$(echo $COMPUTE_CAP | tr -d '.')
    NVCC_GENCODE="-gencode arch=compute_${COMPUTE_CAP_NUM},code=sm_${COMPUTE_CAP_NUM}"
    echo "Detected NVIDIA GPU with compute capability ${COMPUTE_CAP} (sm_${COMPUTE_CAP_NUM})"
  fi
fi

# ffmpeg
cd "$SRC_DIR"
if [ ! -d "ffmpeg" ]; then
  wget -O ffmpeg-snapshot.tar.bz2 https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2 && \
  tar xjvf ffmpeg-snapshot.tar.bz2
fi
cd ffmpeg && \
# Build configure flags
EXTRA_CFLAGS="-I$BUILD_DIR/include -I/usr/local/cuda/include -O3 -march=native -mtune=native"
EXTRA_LDFLAGS="-L$BUILD_DIR/lib -L/usr/local/cuda/lib64 -s"

CONFIGURE_CMD=(
  ./configure
  --prefix="$BUILD_DIR"
  --pkg-config-flags="--static"
  --extra-cflags="$EXTRA_CFLAGS"
  --extra-ldflags="$EXTRA_LDFLAGS"
  --extra-libs="-lpthread -lm"
  --ld="g++"
  --bindir="$BIN_DIR"
  --enable-gpl
  --enable-version3
  --enable-openssl
  --enable-libaom
  --enable-libass
  --enable-libfdk-aac
  --enable-libfontconfig
  --enable-libfreetype
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
  --enable-libzimg
  --enable-libsoxr
  --enable-libsrt
  --enable-vaapi
  --enable-nonfree
  $CUDA_FLAGS
)

if [ -n "$NVCC_GENCODE" ]; then
  CONFIGURE_CMD+=(--nvccflags="$NVCC_GENCODE")
fi

PATH="$BIN_DIR:$PATH" PKG_CONFIG_PATH="$BUILD_DIR/lib/pkgconfig" "${CONFIGURE_CMD[@]}" && \
PATH="$BIN_DIR:$PATH" make -j $NPROC && \
make install && \
hash -r

grep -q "$BUILD_DIR/share/man" "$HOME/.manpath" 2>/dev/null || echo "MANPATH_MAP $BIN_DIR $BUILD_DIR/share/man" >> "$HOME/.manpath"

# rm -rf ~/ffmpeg_build ~/.local/bin/{ffmpeg,ffprobe,ffplay,x264,x265}
# sed -i '/ffmpeg_build/d' ~/.manpath
# hash -r
# --extra-cflags="-D_GNU_SOURCE"
# cat ~/ffmpeg_sources/ffmpeg/ffbuild/config.log
