# netv application image
#
# Default build uses pre-built FFmpeg with full hardware support:
#   docker compose build
#
# Alternative: use apt FFmpeg (fewer codecs, no NVENC/QSV):
#   FFMPEG_IMAGE=ubuntu:26.04 docker compose build
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
# Note: The conditional must be evaluated in shell, not in Dockerfile syntax
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        gosu && \
    # Conditionally install ffmpeg if not present from base image
    if [ ! -x /usr/local/bin/ffmpeg ] && [ ! -x /usr/bin/ffmpeg ]; then \
        apt-get install -y --no-install-recommends ffmpeg; \
    fi && \
    rm -rf /var/lib/apt/lists/*

# Python comes from uv, not the distro, so every base (22.04 ships 3.10) gets the same
# interpreter. The venv is first on PATH, so `python3`/`pip3` resolve to it everywhere.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ARG PYTHON_VERSION=3.14
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN uv venv --seed --python "$PYTHON_VERSION" /opt/venv && python3 --version

# App setup
WORKDIR /app

# Copy application files with verification
COPY pyproject.toml uv.lock README.md ./
COPY netv/ netv/

# Install the locked server dependencies (the versions CI tests) into /opt/venv.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv sync --frozen --no-cache --no-editable --no-default-groups --extra server

# Runtime config
EXPOSE 8000 8100

# Environment variables (see README for details)
ENV NETV_PORT=8000
ENV NETV_GATEWAY_PORT=8100
ENV NETV_HTTPS=""
ENV LOG_LEVEL=INFO
ENV NETV_DATA_DIR=/app/cache

# Match the gateway image so both services can safely share the cache volume.
RUN groupadd --gid 10001 netv && \
    useradd --uid 10001 --gid 10001 --create-home netv

# Copy entrypoint and set permissions with validation
COPY docker/entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh && \
    test -x /app/entrypoint.sh || { echo "ERROR: entrypoint.sh not executable"; exit 1; }

# Healthcheck with improved error handling
# Note: start-period allows time for application startup
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python3 -c "import os, urllib.request; gateway=os.environ.get('NETV_MODE') == 'gateway'; port=os.environ.get('NETV_GATEWAY_PORT', '8100') if gateway else os.environ.get('NETV_PORT', '8000'); path='/healthz' if gateway else '/'; r=urllib.request.urlopen(f'http://localhost:{port}{path}', timeout=5); exit(0 if r.status==200 else 1)" 2>/dev/null || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
