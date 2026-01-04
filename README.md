# neTV

A minimal, self-hosted web interface for IPTV streams.

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

### Consider Alternatives First

If you want a full-featured media center, you'll probably be happier with:

- **[Jellyfin](https://jellyfin.org/)** - Free, open-source media system
- **[Emby](https://emby.media/)** - Media server with IPTV support
- **[Plex](https://plex.tv/)** - Popular media platform with live TV

These are excellent, mature projects with large communities. neTV exists for
users who find them overkill and just want a simple IPTV player.

## Features

- **Live TV** with EPG grid guide
- **Movies & Series** with metadata, seasons, episodes
- **Chromecast** support (HTTPS required)
- **Closed captions** with style customization
- **Search** across all content (supports regex)
- **Favorites** with drag-and-drop ordering
- **Resume playback** for VOD content
- **Responsive** - works on desktop, tablet, mobile
- **Keyboard navigation** - 10-foot UI friendly

### Transcoding

Extensively optimized for minimal latency and CPU usage:

- **Smart passthrough** - h264+aac streams remux without re-encoding (zero CPU)
- **Full GPU pipeline** - NVDEC decode → NVENC/VAAPI encode, CPU stays idle
- **Probe caching** - Streams probed once, series episodes share probe data
- **Interlace detection** - Auto-deinterlaces OTA/cable, skips progressive
- **Smart seeking** - Reuses segments for backward seeks, only transcodes gaps
- **Session recovery** - VOD sessions survive restarts, resume where you left off
- **HTTPS passthrough** - Auto-proxies HTTP streams when behind HTTPS

Hardware transcoding is auto-detected. Check Settings to see available encoders.

- **Intel/AMD (VAAPI)**: Works automatically if `/dev/dri` exists.
- **NVIDIA**: Requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):
  `docker compose --profile nvidia up -d`
- **No GPU / VPS**: If `/dev/dri` doesn't exist, comment out the `devices` section
  in `docker-compose.yml` or compose will fail to start

### Comparison

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
| **Catchup/timeshift** | ❌ | ❌ | ⚠️ Plugin | ⚠️ Plugin | ❌ |
| **Live rewind buffer** | ✅ | ❌ | ⚠️ Via DVR | ⚠️ Via DVR | ⚠️ Via DVR |
| **Resume playback** | ✅ | ❌ | ✅ | ✅ | ✅ |
| **Multi-user** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **User roles** | ⚠️ Admin/viewer | ⚠️ Admin/viewer | ✅ Granular | ✅ Granular | ✅ Granular |
| **Stream limits** | ✅ Per-user, per-source | ❌ | ⚠️ Per-user | ⚠️ Per-user | ⚠️ Per-user |
| **Library permissions** | N/A | N/A | ✅ Per-library | ✅ Per-library | ✅ Per-library |
| **Favorites** | ✅ Drag-and-drop | ✅ | ✅ | ✅ | ✅ |
| **Search** | ✅ Regex | ✅ Basic | ✅ Basic | ✅ Basic | ✅ Basic |
| **Video transcoding** | ✅ | ❌ | ✅ | ✅ | ✅ |
| **Audio transcoding** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Transcode only if needed** | ✅ Auto mode | ❌ | ⚠️ Per-library | ⚠️ Per-library | ⚠️ Per-client |
| **NVENC** | ✅ | ❌ | ✅ | ✅ | ⚠️ Pass |
| **VAAPI** | ✅ | ❌ | ✅ | ✅ | ⚠️ Pass |
| **QSV** | ✅ | ❌ | ✅ | ✅ | ⚠️ Pass |
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

#### Optimized Install (Recommended)

```bash
git clone https://github.com/jvdillon/netv.git
cd netv
docker compose build
docker compose up -d
```

Open http://localhost:8000.

#### Ubuntu Stock Install (Not Recommended)

```bash
git clone https://github.com/jvdillon/netv.git
cd netv
FFMPEG_IMAGE=ubuntu:24.04 docker compose build
docker compose up -d
```

Open http://localhost:8000.

#### Why Prefer Optimized?

| | Optimized | Ubuntu Stock |
|---|---|---|
| FFmpeg source | Pre-built with all codecs | apt (Ubuntu repos) |
| NVENC (NVIDIA) | ✅ | ❌ |
| VAAPI (Intel/AMD) | ✅ | ✅ |
| QSV (Intel QuickSync) | ✅ | ❌ |
| libfdk-aac | ✅ | ❌ |
| SVT-AV1 | ✅ | ❌ |

Verify available encoders: `docker exec netv ffmpeg -encoders | grep -E 'nvenc|vaapi|qsv'`

#### Options

```bash
NETV_PORT=9000 docker compose up -d        # custom port
NETV_HTTPS=1 docker compose up -d          # enable HTTPS (mount certs first)
```

### Local Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/jvdillon/netv.git
cd netv
uv run ./main.py --port 8000  # --https
```

Or with pip:

```bash
pip install .
./main.py --port 8000
```

Open http://localhost:8000, create an admin account, and add your IPTV source.

### Debian/Ubuntu (`systemd`)

For peak FFMPEG performance, Chromecast (requires HTTPS), and auto-start:

```bash
# 1. Install prerequisites (uv, Python)
./tools/install-prereqs.sh

# 2. (Optional) Get HTTPS certificates (required for Chromecast)
./tools/install-letsencrypt.sh yourdomain.com

# 3. (Optional) Build FFmpeg (required for optimal NVidia encoding efficiency)
./tools/install-ffmpeg.sh

# 4. Install systemd service
sudo ./tools/install-netv.sh # default port=8000 or --port 9000 
```

Manage with:

```bash
sudo systemctl status netv       # Check status
sudo systemctl restart netv      # Restart after updates
journalctl -u netv -f            # View logs
sudo systemctl edit netv --full  # Change port or other settings
sudo ./tools/uninstall-netv.sh   # Uninstall
```

There's also some gems in `tools/`:
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

**Where can I get free IPTV?**

Check out [iptv-org/iptv](https://github.com/iptv-org/iptv) -- a community-maintained
collection of publicly available IPTV channels from around the world.

**Where can I get TV guide data?**

The free choice is [iptv-org/epg](https://github.com/iptv-org/epg), but this
has never worked reliably for me.

For a more robust solution, consider [Schedules Direct](https://schedulesdirect.org/) --
your membership helps fund Open Source projects.

Alternatively you can use `tools/zap2xml.py`. I've used this for over a year
and found it to be very reliable -- it scrapes guide data from zap2it/gracenote.

**How do I set up HDHomeRun?**

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

**What are the keyboard shortcuts?**

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

**What Does "neTV" Mean?**

Yes.

We leave pronunciation and meaning as an exercise for your idiom:

- **N-E-T-V** -- "Any TV", say it out loud
- **≠TV** -- "Not Equals TV", because we're `!=` traditional cable
- **Net-V** -- "Net Vision", because it streams video over your network
- **Ni!-TV** -- For the [Knights who say Ni](https://www.youtube.com/watch?v=zIV4poUZAQo)

We will also accept a shrubbery. One that looks nice. And not too expensive.

## License

Apache License 2.0
