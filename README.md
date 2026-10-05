# neTV

A fast, self-hosted IPTV experience for the web, Apple TV, iPhone, macOS,
Chromecast, and Xtream-compatible players.

**What it does:** neTV plays your own Xtream Codes or M3U IPTV subscription
with a clean, minimal interface — live TV with an EPG grid guide, catchup
playback, movies & series, favorites and playlists, Chromecast and AirPlay,
and optional real-time 4K AI upscaling — on desktop, tablet, mobile, and TV.

## Companion Apps

neTV now comes with native companion apps. Install them, point them at your
neTV server, and sign in with your existing neTV account:

- **iPhone, iPad, and Apple TV:** [neTV Personal TV Player](https://apps.apple.com/us/app/netv-personal-tv-player/id6814189386) on the App Store
- **Mac:** [neTV Personal TV Player](https://apps.apple.com/us/app/netv-personal-tv-player/id6814189386?mt=12) on the Mac App Store

The native SwiftUI source for all three apps is in [`apple/`](apple/). See
[Apple Apps](#apple-apps) for details and build instructions.

> **Live TV, upgraded to 4K.** neTV can AI-upscale 1080p streams to 4K with GPU decoding, TensorRT inference, and NVENC
> encoding in one real-time playback pipeline.

> **Missed the start? Go back and press play.** Browse earlier guide listings
> and replay supported upstream archives on the web, Mac, Apple TV, iPhone,
> and iPad -- with full-program seeking and no interruption while browsing.
> [Explore catchup playback](#catchup-playback).

![EPG Guide](screenshots/epg.png)

![Player](screenshots/player.png)

![VOD](screenshots/vod.png)

![Series](screenshots/series.png)

![Settings](screenshots/settings.png)

## Why This Exists

We built neTV because we couldn't find a clean, lightweight interface for
Xtream IPTV services. Existing solutions were either bloated media centers or
clunky apps that didn't work well across devices.

**neTV is intentionally minimal.** It does one thing: play your IPTV streams
with a clean UI that works on desktop, tablet, mobile, and Chromecast.

We also prioritize **keyboard navigation** throughout (though still rough
around the edges). The entire app is theoretically usable with just arrow keys,
Enter, and Escape -- perfect for media PCs, HTPCs, or anyone who prefers
keeping hands on the keyboard (like me).

### Disclaimer

This is a **player only** -- it does not provide any content. You must have your
own IPTV subscription that provides Xtream Codes API access or M3U playlists.
Users are responsible for ensuring they have legal rights to access any content
through their IPTV providers.

## Features

- **Live TV** with EPG grid guide
- **Catchup playback** - Browse **Earlier / Now / Later**, select an available past program, or **Start over** on supported live content. Replay and seek through upstream archives on the web and across the Apple apps. [How it works](#catchup-playback).
- **Mac, Apple TV, and iPhone apps** on the [App Store](https://apps.apple.com/us/app/netv-personal-tv-player/id6814189386)
- **Native player gateway** - Bring your neTV channels and settings to Apple TV and other Xtream-compatible players
- **Movies & Series** with metadata, seasons, episodes
- **Real-time 4K AI Upscale** - 1080p → 4K at 70+ FPS through TensorRT on an RTX 5090
- **GPU-accelerated transcoding** - NVDEC, TensorRT, and NVENC with low-latency HLS
- **Playback resolution badge** - Web and Apple players show the actual video resolution, including 720p, 1080p, and 4K
- **Chromecast** support over your LAN, including HTTP (server-side discovery and controls)
- **AirPlay** from Safari on Mac, iPhone, and iPad to Apple TV and AirPlay 2 TVs
- **Closed captions** from upstream content, with native track selection in every Apple app and style customization on the web
- **Search** across all content (supports regex)
- **Favorites** with drag-and-drop ordering
- **Playlists** - Put your favorite content in playlists and organize them your way. Set them up on the web and they show up everywhere
- **Personalized guide ordering** - Most-played content rises within each category by default
- **Resume playback** for VOD content
- **Responsive** - works on desktop, tablet, mobile
- **Keyboard navigation** - 10-foot UI friendly

### Catchup Playback

**Your guide is now a way back into programs you missed.** When an upstream
keeps an archive, neTV lets you open it directly from the guide -- no separate
recording setup required.

- **Earlier / Now / Later:** browse earlier and upcoming schedules without
  interrupting playback. Apple apps move in three-hour steps, up to seven days
  in either direction. **Now** returns the guide to the current window; it
  does not switch an archive back to live playback.
- **Pick a past program:** select an available listing to play its archive
  from the beginning. Unavailable past listings are dimmed and disabled;
  Apple apps also disable upcoming programs. Guide times use your device's
  time zone.
- **Start over:** restart the program airing now when its upstream archive
  supports it.
- **Seek through the whole program:** drag the timeline on the web, Mac,
  iPhone, or iPad. The web also supports arrow keys and **Jump**; on Apple TV,
  use Left/Right to preview a position and Select or Play to seek.
- **Keep browsing:** playback continues while you explore another guide
  window. On Apple TV, selecting the playing program again expands playback.

Seeking beyond the available video releases the previous upstream connection
before reopening the archive at the selected time, so a seek may briefly
buffer. Catchup **does not use AI upscaling**: it retains the upstream resolution
up to the configured resolution limit. Live playback keeps its existing
upscaling setting.

**Requirements and availability:** configure an Xtream source that advertises
archive support through `tv_archive` and `tv_archive_duration`, with matching
EPG listings. Replay availability depends on the upstream's archive retention;
neTV retains available guide history to cover the longest supported archive,
up to 14 days. It cannot reconstruct missing past listings or create an archive
where none exists. Standalone M3U `catchup="xc"` tags are not currently parsed;
use the upstream's Xtream connection for this feature.

Update the neTV server and rebuild the Apple apps from the current source to
use the new controls. Source support does not imply that an updated App Store
build is already available. See the [Apple app documentation](apple/README.md#catchup-and-archive-seeking)
for native playback details.

### Apple Apps

The [App Store links are at the top of this README](#companion-apps). The Mac
app is a separate client for a neTV server running somewhere else in your
house, such as a media PC, NAS, or home server. Install it, enter your
server's address (for example `http://192.168.1.10:8000`), and sign in with
your neTV account.

The native SwiftUI source for all three apps is in [`apple/`](apple/). To build
them yourself:

```bash
cd apple
cp .env.example .env
# Set NETV_BUNDLE_IDENTIFIER and NETV_DEVELOPMENT_TEAM in .env
xcodegen generate
open neTV.xcodeproj
```

In Xcode, select the iOS, tvOS, or macOS scheme, run it, then sign in with your
existing neTV server URL and account. XcodeGen and Xcode 15 or later are
required.

When upstream content includes a supported text or in-band caption track, neTV
offers it as a selectable native track on Mac, Apple TV, iPhone, and iPad.
Choose **Off** or a named track from the caption button; the choice persists
across playback and quality changes. On Apple TV, press Up from full-screen
playback to focus the caption button and Down to return to playback controls.
The button stays hidden when no source-provided track is available, and
machine-generated transcription is not offered.

### Transcoding

Extensively optimized for minimal latency and CPU usage:

- **Smart passthrough** - h264+aac streams remux without re-encoding (zero CPU)
- **5.1 surround** - Surround audio stays 5.1 (AAC in browsers; Dolby Digital/Digital Plus passes through untouched to the Apple apps)
- **Full GPU pipeline** - NVDEC decode → NVENC/VAAPI encode, CPU stays idle
- **Probe caching** - Streams probed once, series episodes share probe data
- **Interlace detection** - Auto-deinterlaces OTA/cable, skips progressive
- **Smart seeking** - Reuses segments for backward seeks, only transcodes gaps
- **Session recovery** - VOD sessions survive restarts, resume where you left off
- **HTTPS passthrough** - Auto-proxies HTTP streams when behind HTTPS

Apple and web clients share the adaptive live transcoder for configured outputs
above 720p: one upstream ingest, a 720p startup rendition, and a separately warmed
high-quality rendition. Both use the same backend buffer/throughput policy for
upgrades, fallback, and recovery after sustained bandwidth improvement. The high
encoder stays warm during bandwidth saver so recovery reuses the same upstream
ingest. The web player switches Hls.js levels within one media
element using matching playlist dates, and targets 12 seconds behind the live edge
when enough media is available. Native browser HLS without Hls.js telemetry stays
at the safe initial quality.

A server-owned watchdog also checks process state and completed segment publication.
An isolated high-quality failure keeps low playback and the upstream ingest running
while only that encoder restarts. A stalled ingest or low output restarts the
dependency chain only after the old upstream reader exits. Restarted outputs append
to the retained playlists with an HLS discontinuity, preserving the available DVR
window under the same session ID. Recovery attempts have cooldowns and a limit of
three per five-minute window. Workers that exit bypass the publication grace; a
process that remains alive must stop publishing completed media for 45 seconds
before the watchdog intervenes, so temporary upstream delivery gaps do not trigger
a restart.
After a full pipeline recovery, health feedback advances a generation counter so
Apple clients reload the recovered playlist in the same session while preserving
their playback date and pause state when possible.

Live bitrate targets/maximum settings are 4/6 Mbps for 720p, 6/8 Mbps for 1080p,
10/14 Mbps for 1440p, and 16/20 Mbps for 4K, plus audio and transport overhead.
The native gateway's standard live AI-upscale path shares the bitrate policy.
VOD seek/resume, lower-resolution single-rendition playback, and direct playback
retain their existing contracts. All transcoders share HTTP reconnect and process
launching; local inputs do not receive HTTP reconnect flags.

Bulk live/VOD catalog refreshes reuse the per-source loaders used for individual
source refreshes. Bulk refreshes do not rewrite saved EPG URLs, and a failed source
is logged without preventing other sources from loading.

### 4K AI Upscaling

neTV uses NVIDIA TensorRT super-resolution to transform SD and HD channels into
sharp 4K video as they play. The optimized 1080p → 4K pipeline exceeds
**70 FPS on an RTX 5090**, enough for smooth 50/60 FPS live television with
headroom to spare. It also handles compression artifacts, noise, and blur,
making it especially effective for older shows and lower-bitrate streams.

The same pipeline works in the browser and through the native player gateway on
port `8100`, so Apple TV and other Xtream-compatible clients get the same
server-controlled 4K experience.

| Before (720p source) | After (4K AI Upscale) |
|---|---|
| ![Before](screenshots/ai-upscale_price-is-right_disabled.png) | ![After](screenshots/ai-upscale_price-is-right_enabled.png) |
| ![Before](screenshots/ai-upscale_cleopatra_disabled.png) | ![After](screenshots/ai-upscale_cleopatra_enabled.png) |
| ![Before](screenshots/ai-upscale_batman_disabled.png) | ![After](screenshots/ai-upscale_batman_enabled.png) |

Requires an NVIDIA GPU and the [AI Upscale image](#ai-upscale-image-nvidia-gpu).
The Settings page shows AI Upscale options when TensorRT engines are available.

## Alternatives

If you want a full-featured media center, you might be happier with:

- **[Jellyfin](https://jellyfin.org/)** - Free, open-source media system
- **[Emby](https://emby.media/)** - Media server with IPTV support
- **[Plex](https://plex.tv/)** - Popular media platform with live TV

These are excellent, mature projects with large communities. neTV exists for
users who find them overkill and just want a simple IPTV player.

| | neTV | [nodecast-tv] | [Jellyfin] | [Emby] | [Plex] |
|---|---|---|---|---|---|
| **Focus** | IPTV | IPTV | General media | General media | General media |
| **Xtream Codes** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **M3U playlists** | ✅ | ✅ | ✅ | ✅ | ⚠️ Via [xTeVe] |
| **XMLTV EPG** | ✅ | ⚠️ Via provider | ✅ | ✅ | ✅ |
| **Local media** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Live TV** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **VOD (movies/series)** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **DVR recording** | ❌ | ❌ | ✅ | ✅ | ⚠️ Pass |
| **Catchup/timeshift** | ✅ Xtream | ❌ | ⚠️ Plugin | ⚠️ Plugin | ❌ |
| **Live rewind buffer** | ✅ | ❌ | ⚠️ Via DVR | ⚠️ Via DVR | ⚠️ Via DVR |
| **Resume playback** | ✅ | ❌ | ✅ | ✅ | ✅ |
| **Multi-user** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **User roles** | ⚠️ Admin/viewer | ⚠️ Admin/viewer | ✅ Granular | ✅ Granular | ✅ Granular |
| **Stream limits** | ✅ Per-user, per-source | ❌ | ⚠️ Per-user | ⚠️ Per-user | ⚠️ Per-user |
| **Library permissions** | ✅ Per-source | N/A | ✅ Per-library | ✅ Per-library | ✅ Per-library |
| **Favorites** | ✅ Drag-and-drop | ✅ | ✅ | ✅ | ✅ |
| **Search** | ✅ Regex | ✅ Basic | ✅ Basic | ✅ Basic | ✅ Basic |
| **Video transcoding** | ✅ | ❌ | ✅ | ✅ | ✅ |
| **Audio transcoding** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Transcode only if needed** | ✅ Auto mode | ❌ | ⚠️ Per-library | ⚠️ Per-library | ⚠️ Per-client |
| **NVENC** | ✅ | ❌ | ✅ | ✅ | ⚠️ Pass |
| **VAAPI** | ✅ | ❌ | ✅ | ✅ | ⚠️ Pass |
| **QSV** | ✅ | ❌ | ✅ | ✅ | ⚠️ Pass |
| **AI Upscale (4x)** | ✅ TensorRT | ❌ | ⚠️ Plugin | ❌ | ❌ |
| **Software fallback** | ✅ | ❌ Browser | ✅ | ✅ | ✅ |
| **Legacy GPU** | ✅ Any | ❌ No (browser) | ✅ Any | ✅ Any | ⚠️ Driver 450+ |
| **ffprobe caching** | ✅ Dynamic | ❌ None | ⚠️ Offline | ⚠️ Offline | ⚠️ Offline |
| **Episode probe reuse** | ✅ MRU | ❌ No | ⚠️ Per-file | ⚠️ Per-file | ⚠️ Per-file |
| **Session recovery** | ✅ Yes | ❌ No | ⚠️ Via DB | ⚠️ Via DB | ⚠️ Via DB |
| **Auto deinterlace** | ✅ Yes | ❌ No | ⚠️ Manual | ⚠️ Manual | ⚠️ Manual |
| **Subtitles** | ⚠️ WebVTT | ❌ No | ✅ Full | ✅ Full | ✅ Full |
| **Chromecast** | ✅ Yes | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **Keyboard/remote** | ✅ 10-foot UI | ⚠️ Basic | ✅ 10-foot UI | ✅ 10-foot UI | ✅ 10-foot UI |
| **Mobile apps** | ⚠️ Web only | ⚠️ Web only | ✅ Native | ✅ Native | ✅ Native |
| **Subscription** | ✅ Free | ✅ Free | ✅ Free | ⚠️ Premiere | ⚠️ Pass |
| **Setup complexity** | ✅ Minimal | ✅ Minimal | ⚠️ Moderate | ⚠️ Moderate | ⚠️ Moderate |
| **License** | Apache 2.0 | GPL v3 | GPL v2 | GPL v2 | Proprietary |
| **Stack** | Python, FFmpeg | Node.js | .NET, FFmpeg | .NET, FFmpeg | Proprietary |

*Corrections welcome — [open an issue](https://github.com/jvdillon/netv/issues).*

[nodecast-tv]: https://github.com/technomancer702/nodecast-tv
[Jellyfin]: https://jellyfin.org
[Emby]: https://emby.media
[Plex]: https://plex.tv
[xTeVe]: https://github.com/xteve-project/xTeVe

## Installation

### Docker

Create a `docker-compose.yml`:

```yaml
services:
  netv:
    image: ghcr.io/jvdillon/netv:latest
    ports:
      - "8000:8000"
    volumes:
      - ./cache:/app/cache
      - /etc/localtime:/etc/localtime:ro
    devices:
      - /dev/dri:/dev/dri  # for hardware transcoding (remove if no GPU)
    restart: unless-stopped

  netv-gateway:
    image: ghcr.io/jvdillon/netv:latest
    ports:
      - "8100:8100"
    environment:
      - NETV_MODE=gateway
      - NETV_GATEWAY_PORT=8100
    volumes:
      - ./cache:/app/cache
    restart: unless-stopped
```

Then run:

```bash
docker compose up -d
```

Open http://localhost:8000. To update: `docker compose pull && docker compose up -d`

#### Native player gateway

The gateway exposes the live sources configured in the neTV web UI through a
separate Xtream-compatible endpoint. Configure sources and users at
`http://localhost:8000`, then enter the following details in a native IPTV
player:

```text
Server:   http://<netv-host>:8100
Username: <your neTV username>
Password: <your neTV password>
```

The gateway currently supports live categories, live streams, MPEG-TS or HLS
M3U playlists, short EPG lookups, user-filtered XMLTV, and live playback. VOD
and series will be added in later phases.
Remote provider credentials and playback URLs are not returned to the player.

The gateway currently does not enforce the per-user `max_streams_per_source`
setting or coordinate per-source connection counts with the web process. Use a
provider account with sufficient connections and avoid aggressive multi-channel
preview features in native players.

Use a dedicated non-admin neTV account for native players because Xtream
clients include their local credentials in stream URLs.

When running from a source checkout, the gateway has an architecture-neutral
image and can be started independently:

```bash
docker compose up -d --build netv-gateway
```

If neTV is behind a reverse proxy or the address seen by the container is not
reachable by the player, set `NETV_GATEWAY_PUBLIC_URL` to the externally
reachable gateway URL, such as `http://192.168.1.20:8100`.
Use only a scheme, host, and optional port; path prefixes are not supported.
Forwarded client addresses are trusted only from localhost by default. If a
reverse proxy runs elsewhere, set `NETV_GATEWAY_TRUSTED_PROXIES` to its IP
address or a comma-separated list of proxy IP addresses.

For restricted users, the gateway hides uncategorized streams because they
cannot be proven to belong to an allowed category.

Each source can be shared with all users or limited to selected users under
**Settings → Sources → Available To**, so one server can host separate
libraries for different households. Users not selected for a source cannot see
or play its content in the web UI, the Apple apps, or the gateway. Admins
always see every source.

#### Native player 4K upscaling

Port `8100` follows the same AI Upscale, maximum resolution, hardware, and
quality settings configured in the web UI. When an AI model is selected, native
player streams use HLS and run through the TensorRT super-resolution pipeline.
When AI Upscale is disabled, the same endpoint returns MPEG-TS passthrough.

The standard architecture-neutral gateway image supports passthrough only. To
make port `8100` GPU-capable, replace it with the NVIDIA gateway service:

```bash
docker compose stop netv-gateway
docker compose --profile upscale up -d --build netv-gateway-upscale
```

Continue using the same server, username, and password in Apple TV or another
native IPTV player:

```text
Server:   http://<netv-host>:8100
Username: <your neTV username>
Password: <your neTV password>
```

Changes made under **Settings → Transcoding → AI Upscale** apply to new native
player streams without changing the gateway address. Check the active setting
and installed models at `http://<netv-host>:8100/capabilities`.
The first container start builds GPU-specific TensorRT engines and can take a
few minutes.

For a source checkout running directly from `.venv`, the gateway uses the same
default TensorRT engine directory as the web process:
`~/ffmpeg_build/models`. Restart the gateway service after updating the code:

```bash
sudo systemctl restart netv-gateway.service
```

The `latest` image is published on tagged releases, on application changes
merged to `main`, and when a maintainer triggers a rebuild manually. Scheduled
FFmpeg base-image builds do not republish application images, so
`docker compose pull` will not report an update every day.

#### Optional: Nonfree (proprietary) FFMPEG optimized for Nvidia or AMD and/or Intel GPU

We provide a custom built ffmpeg with Nvidia, AMD, and Intel _proprietary
support_ for GPUs. Notably, essential packages are built from source and often
_significantly_ newer than what is baked into Ubuntu 2024 (LTS).

The custom built ffmpeg is not required unless you want:
- best possible GPU performance,
- bleeding edge capability,
- to use AMD discrete GPU,
- realtime AI upscaling (Nvidia only).

Note: the custom built ffmpeg will generally work even if a dependency is not
available. In such cases the specific capability will not be available but
other capabilities will still work. In this sense the custom built ffmpeg is a
"kitchen sink" build.

| | Custom ffmpeg | Ubuntu ffmpeg |
|---|---|---|
| Intel or AMD Integrated GPU (VAAPI) | ✅ | ✅ |
| Intel Integrated GPU (QSV QuickSync) | ✅ | ✅ |
| Nvidia Discrete GPU (NVENC via LLVM) | ❌ | ✅ |
| Nvidia Discrete GPU (NVENC via nvcc) | ✅ | ❌ |
| AMD Discrete GPU (AMF) | ✅ | ❌ |
| Fraunhofer FDK AAC | ✅ | ❌ |
| Realtime AI Upscale (Nvidia TensorRT/Cuda) | ✅ | ❌ |
| AV1 Vulkan | ✅ | ❌ |
| Torch (Nvidia Cuda) | ⚠️ Optional | ❌ |

For Nvidia, you will need the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

To determine which ffmpeg build for Cuda, check your driver and compute capability:
```bash
nvidia-smi --query-gpu=driver_version,compute_cap --format=csv,noheader
# Example: 580.87.02, 8.6 → Driver 580, compute ≥7.5 → use cuda13.0
```

Find your CUDA version ([source](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)):

| Driver | < 7.5 (Maxwell/Pascal/Volta) | ≥ 7.5 (Turing+) |
|--------|------------------------------|-----------------|
| 550 | cuda12.4 | cuda12.4 |
| 560 | cuda12.6 | cuda12.6 |
| 570 | cuda12.8 | cuda12.8 |
| 580+ | cuda12.8 | cuda13.0 |

Then run:
```bash
FFMPEG_IMAGE=ghcr.io/jvdillon/netv-ffmpeg:<cuda-version> docker compose --profile nvidia up -d
```

For AMD or Intel, it does not matter which version you choose nor do you need Cuda installed.

#### AI Upscale Image (Nvidia GPU)

For real-time AI upscaling, including **1080p → 4K at 70+ FPS** on an RTX 5090:

```bash
git clone https://github.com/jvdillon/netv.git
cd netv
docker build -f Dockerfile.ai_upscale -t netv-ai-upscale .
docker run --gpus all -v netv-models:/models -v ./cache:/app/cache -p 8000:8000 netv-ai-upscale
```

First start builds TensorRT engines for your GPU (~2-3 min). Engines are cached in the
`netv-models` volume for instant subsequent starts.

<details>
<summary>AI model attribution</summary>

The recommended 1080p upscaler is based on
[`Phips/2xNomosUni_compact_otf_medium`](https://huggingface.co/Phips/2xNomosUni_compact_otf_medium)
by Philip Hofmann and is used under the CC BY 4.0 license.

</details>

Requirements:
- Nvidia GPU (RTX 20xx or newer recommended)
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Driver 535+ (CUDA 12.x)

#### Docker Custom Builds

For customization or development:

```bash
git clone https://github.com/jvdillon/netv.git
cd netv
docker compose build                              # optimized FFmpeg (default)
# FFMPEG_IMAGE=ubuntu:24.04 docker compose build  # or stock FFmpeg
docker compose up -d
```

To update: `git pull && docker compose build && docker compose up -d`

#### Options

```bash
NETV_PORT=9000 docker compose up -d        # custom port
NETV_HTTPS=1 docker compose up -d          # enable HTTPS (mount certs first)
```

### Chromecast on a local network

Open neTV using the server's LAN address (for example,
`http://192.168.1.10:8000`), start a channel, movie, or episode, and select
**Cast to TV** in the player. Select a discovered device or enter the
Chromecast's LAN IP manually. The server controls the TV, so this also works
from iPhone browsers without Google's browser Cast SDK. No domain, HTTPS
certificate, or internet-facing port forwarding is required.

Only **Google Cast** receivers are supported: Chromecast, Google TV/Android TV,
Nest Hub, and TVs with Chromecast built-in. Apple TV and AirPlay-only TVs
(including many LG and Samsung models) are not Cast receivers and will not be
listed; use [AirPlay from Safari](#airplay-from-safari) instead. A "cast" button in YouTube or Netflix on those TVs uses DIAL, not
Google Cast. Audio-only Cast speakers (Nest Mini, cast-enabled receivers) may be
discovered but cannot display video.

**neTV address reachable by the TV** must include the correct protocol and
published port. It defaults to the browser's address and is remembered per user
after a successful cast. Replace `localhost`, a Docker-only address, or an
unreachable reverse-proxy hostname with the server's LAN address. If using HTTPS,
the TV must trust its certificate; accepting a self-signed certificate in your
browser does not make the TV trust it.

Casting uses neTV's HLS transcode/remux pipeline, even when browser playback was
direct. FFmpeg must be installed, and the selected output codecs/resolution must
be supported by the receiver. Live casting starts at the live edge; movies and
episodes transfer the current position. Cast controls provide play, pause,
volume, and stop. Movie/episode watch positions are saved by the server.
Seeking and automatic next-episode playback on the TV are not currently supported.

The browser pauses after the receiver accepts the stream. You can close the page
and playback continues; return to a player page to control the active cast.
Each user can control one cast at a time, and a TV already casting for another
neTV user cannot be taken over. **Stop casting and play here** returns playback
to the browser. Restarting neTV ends its casting sessions.

**Docker/networking:** default bridge networking can prevent mDNS discovery.
Manual TV IP entry works without multicast when the container can reach the TV
and the TV can reach neTV's published port. On Linux, an optional host-network
override enables LAN discovery (Docker Compose 2.24+):

```bash
docker compose -f docker-compose.yml -f docker-compose.cast.yml up -d --build netv
```

With this override, `NETV_PORT` is the actual listening port; Docker port mappings
are removed. Host networking is Linux-specific here; on Docker Desktop, prefer
manual IP entry. Allow server-to-TV Cast traffic (TCP 8009 and device information
on 8008/8443), TV-to-neTV HTTP traffic, and UDP 5353 multicast for discovery.
Guest Wi-Fi/client isolation or separate VLANs can block these paths. This does
not make the Chromecast fully offline: its receiver software may still need
internet access.

### AirPlay from Safari

In Safari on macOS, iOS, or iPadOS, open neTV using the server's LAN address
(for example, `http://192.168.1.10:8000`, not `localhost`), start playback, and
select the **AirPlay** button in the player. It sends the stream to an Apple TV
or an AirPlay 2 TV (such as LG, Samsung, Sony, or Vizio models). No HTTPS
certificate is required. The button is shown only in Safari; other browsers
should use Chromecast.

AirPlay switches the player from hls.js to Safari's native HLS player, because
hls.js disables remote playback. If the stream was playing directly from the
provider, the first tap prepares a neTV HLS stream; tap **AirPlay** again to
choose the TV. The TV fetches the stream from neTV itself, so it must be able to
reach neTV's port on your LAN. Keep the Safari tab open while AirPlaying.

### Debian/Ubuntu (`systemd`)

For peak FFMPEG performance and auto-start:

```bash
# 1. Install prerequisites (uv, Python)
./tools/install-prereqs.sh

# 2. (Optional) Get HTTPS certificates (not required for Chromecast)
./tools/install-letsencrypt.sh yourdomain.com

# 3. (Optional) Build FFmpeg (required for optimal Nvidia encoding efficiency)
./tools/install-ffmpeg.sh

# 4. (Optional) Build AI Upscale engines (requires Nvidia GPU + TensorRT)
uv sync --group ai_upscale
./tools/install-ai_upscale.sh

# 5. Install systemd service
sudo ./tools/install-netv.sh # default port=8000 or --port 9000
```

The default installer builds the recommended engines for both SD/720p and
1080p sources.

After the engine is built, open **Settings → Transcoding**, select
the recommended 1080p option under **AI Upscale**, choose a 4K maximum
resolution, and set transcoding to **Always**. New web and port `8100` streams
use the setting without restarting the services.

Manage with:

```bash
sudo systemctl status netv       # Check status
sudo systemctl restart netv      # Restart after updates
journalctl -u netv -f            # View logs
sudo systemctl edit netv --full  # Change port or other settings
sudo ./tools/uninstall-netv.sh   # Uninstall
```

### Development/Testing

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and FFmpeg (`ffmpeg`
and `ffprobe` on `PATH`, e.g. `brew install ffmpeg` or `apt install ffmpeg`).
Without FFmpeg, streams fail to start:

```bash
git clone https://github.com/jvdillon/netv.git
cd netv
uv run ./main.py --port 8000  # --https
uv run ./gateway.py --port 8100
```

Or with pip:

```bash
pip install .
./main.py --port 8000
```

Open http://localhost:8000, create an admin account, and add your IPTV source.

Web playback regression checks use Node's built-in runner (Node 18+):

```bash
node --test collector_tests/*.test.cjs
```

### Additional Gems

There's also some useful applications in `tools/`:
- `zap2xml.py`: Scrape guide data into XML (I `crontab` this at 5am daily).
- `alignm3u.py`: Useful for reworking your HDHomeRun m3u to align with guide.
- `xtream2m3u.py`: Dump xtream to m3u, useful for making Emby work with IPTV.

## Troubleshooting

### Debug Logging

Enable verbose logs to diagnose EPG, M3U parsing, or other issues.

**Docker:**

In `docker-compose.yml`, change `LOG_LEVEL=INFO` to `LOG_LEVEL=DEBUG`, then restart:

```bash
docker compose down && docker compose up -d
docker compose logs -f
```

**Systemd:**

```bash
sudo systemctl edit netv
```

Add:

```ini
[Service]
Environment="LOG_LEVEL=DEBUG"
```

Then restart and view logs:

```bash
sudo systemctl restart netv
journalctl -u netv -f
```

**Manual / Development:**

```bash
LOG_LEVEL=DEBUG ./main.py
# or
./main.py --debug
```

## Q&A

### Where can I get free IPTV?

Check out [iptv-org/iptv](https://github.com/iptv-org/iptv) -- a community-maintained
collection of publicly available IPTV channels from around the world.

### Where can I get TV guide data?

The free choice is [iptv-org/epg](https://github.com/iptv-org/epg), but this
has never worked reliably for me.

For a more robust solution, consider [Schedules Direct](https://schedulesdirect.org/) --
your membership helps fund Open Source projects.

Alternatively you can use `tools/zap2xml.py`. I've used this for over a year
and found it to be very reliable -- it scrapes guide data from zap2it/gracenote.

### How do I set up HDHomeRun?

HDHomeRun devices provide an M3U playlist, but it lacks EPG channel IDs. Use the
`tools/` to fetch guide data and align it:

```bash
# 1. Get your HDHomeRun lineup (replace IP with your device's IP)
wget http://192.168.1.87/lineup.m3u -O tools/lineup.m3u

# 2. Fetch TV guide data for your area
./tools/zap2xml.py --zip 90210

# 3. Align the M3U with the guide (adds tvg-id for EPG matching)
./tools/alignm3u.py --input tools/lineup.m3u --xmltv tools/xmltv.xml --output tools/ota.m3u
```

Then add `tools/ota.m3u` as an M3U source in neTV settings.

And set up a cron job to refresh the guide daily (e.g.,
`0 5 * * *  /usr/bin/python3 /path/to/netv/tools/zap2xml.py --zip 90210 && cp /path/to/netv/tools/xmltv.xml /var/www/html/`).

### How do I enable hardware transcoding?

Hardware transcoding is auto-detected. Check Settings to see available encoders.

- **Intel/AMD (VAAPI)**: Works automatically if `/dev/dri` exists.
- **Nvidia**: Requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  See [Nvidia GPU (NVENC)](#nvidia-gpu-nvenc) installation section for driver/compute compatibility table.
- **No GPU / VPS**: If `/dev/dri` doesn't exist, comment out the `devices` section
  in `docker-compose.yml` or compose will fail to start

### How do I install CUDA on Ubuntu?

Tested on Ubuntu 24.04 LTS, 25.04, and 25.10.

```bash
# Step 1: Remove existing Nvidia packages
sudo apt purge -y '^nv.*' '^libnv.*' '^cuda-.*' '^libcuda-.*' '^cudnn[0-9]*-.*' '^libcudnn[0-9]*-.*'
sudo apt autoremove -y

# Step 2: Add Nvidia CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt modernize-sources || true
sudo apt update

# Step 3: Install driver and CUDA toolkit
# For Turing+ GPUs (RTX 20 series and newer, compute >=7.5):
sudo apt install -y nvidia-open cuda-toolkit-13 cudnn9-cuda-13 libcudnn9-dev-cuda-13 libnvinfer-bin

# For Maxwell/Pascal GPUs (GTX 900/1000 series, compute <7.5):
# Driver 590 dropped support. Pin to 580 and use CUDA 12.8.
# Note: Maxwell/Pascal requires nvidia-driver (proprietary), not nvidia-open.
# sudo apt install -y nvidia-driver-pinning-580
# sudo apt install -y nvidia-driver-580 cuda-toolkit-12-8 cudnn9-cuda-12-8 libcudnn9-dev-cuda-12 libnvinfer-bin
# sudo update-alternatives --set cuda /usr/local/cuda-12.8

# Step 4: Configure environment
tee -a ~/.bashrc << 'EOF'
export CUDA_HOME=/usr/local/cuda
if [ -d $CUDA_HOME ]; then
    export PATH="${CUDA_HOME}/bin${PATH:+:${PATH}}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
unset CUDA_HOME
EOF
source ~/.bashrc

# Step 5: Verify installation
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
nvcc --version
```

### What are the keyboard shortcuts?

| Key | Action |
|-----|--------|
| `Space` / `k` | Play/pause |
| `f` | Fullscreen |
| `m` | Mute |
| `c` | Toggle captions |
| `i` | Toggle info overlay |
| `←` / `→` | Seek ±10s |
| `↑` / `↓` | Volume |
| `j` | Jump to time |
| `Esc` | Back / close |

### What Does "neTV" Mean?

Yes.

We leave pronunciation and meaning as an exercise for your idiom:

- **N-E-T-V** -- "Any TV", say it out loud
- **≠TV** -- "Not Equals TV", because we're `!=` traditional cable
- **Net-V** -- "Net Vision", because it streams video over your network
- **Ni!-TV** -- For the [Knights who say Ni](https://www.youtube.com/watch?v=zIV4poUZAQo)

We will also accept a shrubbery. One that looks nice. And not too expensive.

## Support

If you find neTV useful, consider buying me a coffee:

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=flat&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/jvdillon)

## License

Apache License 2.0
