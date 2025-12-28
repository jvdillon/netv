# neTV

A minimal, self-hosted web interface for IPTV streams.

![EPG Guide](screenshots/epg.png)

![Player](screenshots/player.png)

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
- **Hardware transcoding** (NVIDIA NVENC, VAAPI, QSV)
- **Closed captions** with style customization
- **Search** across all content (supports regex)
- **Favorites** with drag-and-drop ordering
- **Resume playback** for VOD content
- **Responsive** - works on desktop, tablet, mobile
- **Keyboard navigation** - 10-foot UI friendly

## Disclaimer

This is a **player only** -- it does not provide any content. You must have your
own IPTV subscription that provides Xtream Codes API access or M3U playlists.
Users are responsible for ensuring they have legal rights to access any content
through their IPTV providers.

## Quick Start

```bash
git clone https://github.com/jvdillon/netv.git
cd netv
chmod +x main.py
./main.py --port 8000  # --https
```

Open `http://localhost:8000`, create an admin account, and add your IPTV source.

## Installation

### Debian/Ubuntu

For peak FFMPEG performance, Chromecast (requires HTTPS), and auto-start:

```bash
# 1. Install prerequisites (uv, Python)
./tools/install-prereqs.sh

# 2. Get HTTPS certificates
./tools/install-letsencrypt.sh yourdomain.com

# 3. (Optional) Build FFmpeg with hardware encoding
./tools/install-ffmpeg.sh

# 4. Install systemd service
sudo ./tools/install-netv.sh
```

Manage with:

```bash
sudo systemctl status netv      # Check status
sudo systemctl restart netv     # Restart after updates
journalctl -u netv -f           # View logs
```

Theres also some gems in `tools/`:
- `zap2xml.py`: Scrape guide data into to XML (I `crontab` this at 5am daily).
- `alignm3u.py`: Useful for reworking your HDHomeRun m3u to align with guide.
- `xtream2m3u.py`: Dump xtream to m3u, useful for making Emby work with IPTV.

## Adding Sources

In Settings, add your IPTV source:

- **Xtream Codes**: Server URL, username, password from your provider
- **M3U**: Playlist URL from your provider

## Configuration

Settings are stored in `cache/settings.json`:

| Setting | Values | Description |
|---------|--------|-------------|
| `transcode_mode` | `auto`, `always`, `never` | When to transcode streams |
| `transcode_hw` | `nvidia`, `vaapi`, `qsv`, `software` | Hardware encoder |
| `captions_enabled` | `true`, `false` | Default caption state |

## Hardware Transcoding

Transcoding converts streams your browser can't play natively. Check available
encoders:

```bash
ffmpeg -encoders | grep -E 'nvenc|vaapi|qsv'
```

If empty, your FFmpeg lacks hardware support. Distribution packages often
exclude these due to licensing. Build from source:

```bash
./tools/install-ffmpeg.sh
```

This compiles FFmpeg with NVENC, VAAPI, and non-free codecs (libfdk-aac, x264,
x265, AV1).

## Keyboard Shortcuts

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

## Security

- Change the default password after first login
- Use HTTPS for Chromecast and remote access
- The app binds to `0.0.0.0` -- use a firewall to restrict access

## What Does "neTV" Mean?

Yes.

We leave pronunciation and meaning as an exercise for your idiom:

- **N-E-T-V** -- "Any TV", say it out loud
- **≠TV** -- "Not Equals TV", because we're `!=` traditional cable
- **Net-V** -- "Net Vision", because it streams video over your network
- **Ni!-TV** -- For the [Knights who say Ni](https://www.youtube.com/watch?v=zIV4poUZAQo)

We will also accept a shrubbery. One that looks nice. And not too expensive.

## License

Apache License 2.0
