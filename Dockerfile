# netv application image
#
# Default build uses pre-built FFmpeg with full hardware support:
#   docker compose build
#
# Alternative: use apt FFmpeg (fewer codecs, no NVENC/QSV):
#   FFMPEG_IMAGE=ubuntu:24.04 docker compose build
#
# The optimized FFmpeg base image includes:
# - NVENC (NVIDIA hardware encoding)
# - VAAPI (Intel/AMD hardware encoding)
# - QSV/VPL (Intel QuickSync)
# - All major codecs (x264, x265, VP9, AV1, etc.)

ARG FFMPEG_IMAGE=ghcr.io/jvdillon/netv-ffmpeg:latest
FROM ${FFMPEG_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
# - If using apt ffmpeg (ubuntu base): install ffmpeg + python
# - If using compiled ffmpeg (netv-ffmpeg base): ffmpeg already present, just install python
RUN apt-get update && apt-get install -y --no-install-recommends \
    gosu \
    python3 \
    python3-pip \
    $(if [ ! -f /usr/local/bin/ffmpeg ]; then echo "ffmpeg"; fi) \
    && rm -rf /var/lib/apt/lists/*

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

ENV NETV_PORT=8000
ENV NETV_HTTPS=""
ENV LOG_LEVEL=INFO

# Create non-root user (entrypoint handles permissions and group membership)
RUN useradd -m netv

# Copy entrypoint and set permissions
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Healthcheck (internal port is always 8000)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/', timeout=5)" || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
