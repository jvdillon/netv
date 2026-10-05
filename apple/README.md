# neTV for Apple

Native SwiftUI clients for iPhone, iPad, Apple TV, and Mac. All apps share the same authenticated neTV guide, channel artwork, program metadata, and AVPlayer playback while adapting navigation and layout to each platform.

Mac and Apple TV share the same live layout and slate/purple color scheme, with
larger text and controls on TV. A narrow Live TV / Settings navigation rail sits
beside a full-height category column with its own scrollbar. Categories stay
visible while the adjacent channel list scrolls, and a single click switches
categories locally without waiting for a network request. Category names appear
only once: names differing only in case or surrounding whitespace are grouped,
and selecting a group includes channels from all of its underlying category IDs.
The first occurrence determines the displayed name and order.

The guide uses compact logo-led channel rows, purple current-program highlights,
half-hour time markers, and a now indicator. A small video preview occupies the
top-right corner, with the channel, program title, times, and description beside
it across the top; the channel guide is below. Selecting a category only filters
the list; selecting a channel tunes playback. Search covers channel names and all
loaded program titles, and channel counts follow the search.

On TV, the navigation rail also has Search and Refresh controls. Select a channel
once to play it in the preview, then select the same channel again to fill the
display. Back/Menu returns to the guide and focuses the playing channel, so
pressing Select again returns to fullscreen without restarting playback. Video
retains its original aspect ratio. In fullscreen, the remote's Play/Pause button
or a click on the touchpad pauses and resumes. Left and Right seek through the
retained live window in 15-second steps without requiring a pause. The status bar
spans the scheduled program from start to end. A white marker shows the playback
position, a red marker shows the live edge, and the differently shaded gap between
them shows how far playback is behind. The bar stays visible while paused. When
guide timestamps are unavailable, it falls back to the retained-window timeline.
The server retains everything watched in the current session, up to the configured
two-hour limit, without replacing the live session.

The scheduled-program timeline and 15-second live seek controls are shared by
Apple TV, Mac, iPhone, and iPad. Mac shows its control bar on hover and keeps it
visible while paused; iPhone and iPad use touch controls below the same timeline.
The resolution badge appears only with these controls, including while paused or
seeking. AVKit's native transport controls are not used. TV has no fullscreen
button or percentage/volume panel; use the Siri Remote's hardware volume keys to
control the connected display or receiver. Mac retains its fullscreen button.
App volume carries across playback changes and adaptive quality switches.

Category names and ordering follow the web settings. Use the updated neTV server
for category metadata and device-local program times; older servers remain
playable through All Channels. The Apple client loads every guide page rather
than stopping at the first 500 channels. Use the refresh button to reload the
guide data and three-hour schedule. The guide also reloads and re-anchors to
the current window whenever it comes back into view, including after Settings,
fullscreen playback, or returning to the app. Existing rows remain interactive
while this refresh runs so remote navigation is not interrupted.

## Catchup and archive seeking

Use the guide's **Earlier**, **Now**, and **Later** buttons on every Apple platform
to browse three-hour windows, up to seven days in either direction. The date and
times use your device's time zone. Browsing does not interrupt playback.
Select an available past program directly to play its upstream archive; unavailable
past programs and upcoming programs are dimmed and cannot be selected.
Stream names and logos are noninteractive labels; choose a program to play.
When no listings are available, a **Watch live** tile in the guide keeps live
playback accessible. **Now** returns the guide to the current window without
changing what is playing.

On Apple TV, focus a program and press Select to play it; select the playing program
again to expand it. Moving left out of the program grid skips the stream labels
and returns to categories. The time controls and program grid form one vertical
focus path, so Up and Down move between them from any control. Leaving fullscreen
restores focus to an available program rather than the stream label. Program context
menus remain available for **Start Over** and **Catch Up…**, but is no longer required
to browse earlier programs.
On iPhone, iPad, and Mac, drag the archive timeline
to preview a position and release to seek, or use the ten-second skip buttons.
The timeline covers the full program, including portions not yet buffered.
Seeking preserves whether playback was paused.

On Apple TV, expand archive playback and press Left/Right to preview a position
in ten-second steps. Press Select or Play/Pause to commit the seek and play.
No new upstream request is made while previewing.

Positions already available in the current HLS recording seek locally. Other
positions reopen the upstream archive at the selected time, waiting for the old
session to stop first. The player accounts for the archive's minute-aligned start
and keeps the session alive while paused. Catchup uses source resolution (up to
the server's configured limit), without AI upscaling; live playback is unchanged.

## AirPlay

On iPhone, iPad, and Mac, the player has an **AirPlay** button (top right on
iPhone/iPad, in the hover control bar on Mac). It sends video to an Apple TV or an AirPlay 2
TV, which fetches the stream from neTV itself. Sign in with the server's LAN
address, such as `http://192.168.1.10:8000`: a TV cannot reach `localhost`, and
the Mac app warns when AirPlay is active with a loopback server address. While
AirPlaying, the app keeps the current quality rather than switching renditions,
because replacing the player would drop the AirPlay route.

tvOS does not let apps AirPlay video from an Apple TV to another screen. To send
Apple TV audio to AirPlay speakers, use the system Control Center.

## Open and run

1. Copy `apple/.env.example` to `apple/.env`, then set your bundle identifier and Apple development team. The local `.env` is ignored by Git.
2. Generate the project with `cd apple && xcodegen generate`.
3. Open `apple/neTV.xcodeproj`.
4. Select **neTV-iOS**, **neTV-tvOS**, or **neTV-macOS**, choose a destination, and run.
5. Sign in with the same neTV server address and account used by the web UI.

Signing is managed automatically for all three targets through `Config.xcconfig`.
For TestFlight or App Store distribution, sign in to Xcode with an Apple ID that
can access the team configured in `.env`.

To create a tvOS TestFlight archive from the repository root, run:

```bash
./apple/archive-testflight.sh
```

The script assigns a unique UTC build number in `YYMM.DDHH.MMSS` format, creates
a signed Release archive, and opens it in Xcode Organizer. Use Organizer's
**Distribute App** workflow to upload it. Set `NETV_BUILD_NUMBER` to override the
generated build number or `NETV_ARCHIVE_PATH` to choose another archive path.
Automatic signing still requires account access and valid distribution signing
credentials; it does not replace them.

The development default is `http://localhost:8000`. HTTP transport is enabled because neTV commonly runs on a trusted local network; production deployments should use HTTPS.

### Local network access on iPhone and iPad

Allow neTV to access the local network when iOS asks. API requests do not wait
for connectivity, so unavailable network access is reported promptly rather
than holding the startup screen. Retry sign-in after granting permission.
Requests still have a 60-second overall timeout.

If sign-in reports no network access, check **Settings > Privacy & Security >
Local Network > neTV**, and make sure your Wi-Fi or VPN can reach the server.
Safari opening the same address does not prove that neTV has permission:
[Safari is exempt from local-network privacy checks](https://developer.apple.com/documentation/technotes/tn3179-understanding-local-network-privacy).
If neTV is missing from that list and no prompt appears, restart the device,
reopen the current app build, and retry sign-in with the app in the foreground.
The app cannot grant or reset this system permission itself.

## Shared adaptive live playback

The Apple app and web player request the same fast-start backend for transcoded
live channels configured above 720p. One FFmpeg ingest reads the provider and
remuxes a rolling local feed; a
720p encoder builds the startup reserve first, then an independent encoder starts
preparing the configured quality, including AI upscaling. This keeps high-quality
GPU initialization out of initial buffer preparation, at the cost of a later
quality upgrade. Playback does not wait for high-quality output.
Both encoders read local files, so warming up or changing quality does not open
another provider stream. This requires capacity
for two local encoders and up to roughly two minutes of source segments on disk
(source keyframe spacing can lengthen that window).

The 720p rendition targets 4 Mbps with a 6 Mbps peak setting. The high rendition
uses the same shared bitrate policy: 1080p targets 6 Mbps with an 8 Mbps
peak setting, 1440p targets 10/14 Mbps, and 4K targets 16/20 Mbps. This replaces
unbounded constant-QP encoding for adaptive live playback, trading compression quality
for predictable bandwidth. Audio and transport overhead are additional; the
upgrade decision still uses measured segment sizes rather than assuming the
encoder setting is an exact network cap. NVENC uses
[variable bitrate control](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/ffmpeg-with-nvidia-gpu/index.html).
The standard live AI-upscale encoder used by the native gateway shares these
limits too; VOD encoding retains its existing quality settings.

Ingest limits initial FFmpeg analysis to one second of media and starts the local
720p encoder as soon as one complete source segment is available, rather than waiting
for a second source keyframe interval before encoder warm-up. It retains probed
packets; `nobuffer` is deliberately not used because dropping those packets can
delay the first decodable frame. Logs report elapsed time to input readiness and
720p readiness separately.

Playback still waits for at least eight seconds of 720p media (or two source segment
durations, whichever is longer) to bridge bursty upstream delivery. Each rendition
keeps at least the normal 30-second live window. The app requests a 12-second
forward buffer. Apple channel changes and in-page web restarts wait for any earlier
startup to return its session ID, then confirm that session has stopped before
starting the next provider feed. The backend checks for disconnected requests while
waiting for adaptive startup and cleans up their processes and files.

The web player consumes a local master playlist containing the same low/high
renditions. Hls.js starts pinned to low quality, reports buffer and recent fragment
download measurements every two seconds, and follows the backend's health decisions
using in-place level switching. Both playlists carry matching program dates; the
web player does not reload the source or open another provider stream on upgrades.
It targets a 12-second live delay when enough media is available, rather than
treating transcoded live streams as VOD. Native browser HLS without Hls.js telemetry
stays on the initial rendition and only sends heartbeats.

An upgrade requires fresh high-quality segments caught up to the low rendition,
at least six seconds of player buffer, and three download measurements spanning
at least six seconds with 50% headroom over the largest recent high-quality
segment bitrate. Measurements expire after ten seconds; polls with no download
do not erase recent evidence. Encoder readiness is checked when switching rather
than requiring continuous alignment throughout each source delivery burst.
Unavailable throughput leaves playback at 720p. Shared MPEG-TS timestamps produce matching
playlist dates; the app prepares the replacement while playback continues and
seeks to the current broadcast time before switching. A brief buffering pause is
still possible. A failed high-quality encoder leaves 720p available.

For live playback using neTV's transcoder (`transcode_mode: always`), the Apple app
reports buffer level, waiting state, and recent download throughput every two seconds.
After a 15-second startup grace period, the backend requests bandwidth saver if the
buffer remains below three seconds and playback is waiting or downloads cannot keep
up for eight seconds. Pauses and brief stalls do not trigger it.

Fast-start sessions switch back to their local 720p rendition while keeping the
high encoder running on the same local ingest. This uses more server CPU/GPU than
stopping that encoder, but allows recovery without reopening the provider stream.
The server holds bandwidth saver for at least 60 seconds and requires 30 seconds
of healthy playback with at least eight seconds buffered and download throughput
at least 50% above the high rendition's bitrate. Download measurements must stay
fresh (no gaps over ten seconds), and at least three measurements are required.
The high encoder must also have fresh, aligned segments before recovery. Pauses,
stalls, insufficient throughput, and long telemetry gaps reset recovery evidence.

The Apple app follows both fallback and recovery decisions, carrying the current
decision across channel changes rather than latching saver until restart. New
fast-start channels still prepare both renditions when saver is enabled. Legacy
single-rendition sessions stop and retune on either quality change, releasing the
old session first; a brief interruption is expected. Their recovery threshold uses
the configured high bitrate plus transport overhead and any larger requirement
observed before fallback. Their reduced mode disables AI upscaling and uses a 720p
maximum (480p if already configured) and the low encoder quality preset. Global
server settings are unchanged. LTE and dropped-frame counts alone do not trigger
fallback. Direct/passthrough streams do not use this backend feedback mechanism.
After a full fast-start pipeline recovery, the app reloads the new generation in
the same session and preserves its playback date and pause state when possible.

An upstream advertising `m3u8` supports HLS, but not necessarily adaptive bitrate.
Confirm multiple `#EXT-X-STREAM-INF` variants in a channel's master playlist before
assuming provider-side adaptation is available. A media playlist with only
`#EXTINF` segments is a single rendition. Inspect provider playlists only when a
connection slot is free: even a playlist request can count as a stream on
single-connection accounts. The fast-start path creates its two qualities locally;
it does not implement provider-side adaptive bitrate switching. Its ingest already
avoids a separate `ffprobe` connection regardless of the `probe_live` setting.

To check on a device, play a transcoded live channel and throttle the connection
until it repeatedly runs out of buffer. Check that playback returns to 720p
without a second provider connection, then restore bandwidth and verify recovery
after the cooldown and sustained-health requirements. Confirm the session ID and
provider ingest stay unchanged through both switches. Also check that a short
stall or manual pause does not cause a retune, and that tuning another channel
starts with the current saver decision and can subsequently recover. Both the
server and Apple app must be updated for the complete recovery behavior.

For backend testing on `aitony.tulane`, use the updated Apple app with this branch
on the server, or reload the web player after restarting the updated backend.
Confirm startup at 720p, upgrade on a healthy connection, continued
720p while the high encoder is unavailable, and cleanup of all three processes
when playback stops. The backend fast-start path is selected by both clients through
`/transcode/start?fast_start=true`; older clients and gateway players retain their existing
playback contracts. The response includes `playlist` for the initial rendition and
`master_playlist` for clients that switch HLS levels in place.
The service logs `Playback quality` every ten seconds, including the decision
reason, buffer, measured throughput, high-quality bitrate, and alignment state.

The cancellation/rapid-tuning regression check uses a fake transport with the real
app model. On a Mac, compile and run it with:

```sh
xcrun swiftc -parse-as-library Shared/Models.swift Shared/AppModel.swift Tests/PlaybackStartChecks.swift -o /tmp/netv-playback-start-checks
/tmp/netv-playback-start-checks
```

The guide decoding and timeline checks run separately:

```sh
xcrun swiftc -parse-as-library Shared/Models.swift Tests/GuideDataChecks.swift -o /tmp/netv-guide-checks
/tmp/netv-guide-checks
```

Archive metadata, heartbeat, and forced-release requests use the real API client
with an in-memory HTTP transport:

```sh
xcrun swiftc -parse-as-library Shared/Models.swift Shared/APIClient.swift Tests/ArchiveAPIChecks.swift -o /tmp/netv-archive-api-checks
/tmp/netv-archive-api-checks
```
