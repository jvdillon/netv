import AVKit
import Combine
#if os(iOS)
import AVFAudio
#endif
import OSLog
import SwiftUI

struct PlayerView: View {
    @EnvironmentObject private var model: AppModel
    let selection: PlayerSelection

    @State private var player: AVPlayer?
    @State private var errorMessage: String?
    @State private var quality: String?
    @State private var airPlayActive = false
    @State private var airPlayHost: String?
    @State private var archiveStreamStart: Double?
    @State private var isScrubbing = false
    @State private var seekError: String?
    @State private var seekTask: Task<Void, Never>?
    @State private var isPreparingArchive = false
    @State private var liveBufferDuration = 0.0
    #if os(tvOS)
    @State private var tvActivity = Date()
    @State private var tvSeekPosition: Double?
    @State private var tvLiveSeekPosition: Double?
    @State private var tvLiveResumeAfterSeek = false
    @State private var tvResumeAfterCancel = false
    #endif
    private let logger = Logger(
        subsystem: Bundle.main.bundleIdentifier ?? "com.netv",
        category: "Player"
    )

    private var archiveTimeline: ArchiveTimeline? {
        ArchiveTimeline(selection: selection, streamStart: archiveStreamStart)
    }

    private var liveProgram: Program? {
        guard !selection.isCatchup else { return nil }
        if let row = model.channels.first(where: { $0.id == selection.channel.id }),
           let program = row.currentProgram {
            return program
        }
        return selection.program?.isCurrent == true ? selection.program : nil
    }

    var body: some View {
        ZStack {
            Color.black
            if let player {
                PlayerController(player: player, isCatchup: selection.isCatchup)
            } else if let errorMessage {
                ContentUnavailableView(
                    "Unable to Play",
                    systemImage: "exclamationmark.triangle",
                    description: Text(errorMessage)
                )
            } else {
                ProgressView("Tuning \(selection.channel.name)…")
                    .tint(.white)
            }
            if player != nil && isPreparingArchive {
                ProgressView("Seeking…")
                    .padding()
                    .background(.black.opacity(0.7), in: RoundedRectangle(cornerRadius: 12))
                    .tint(.white)
            }

            #if os(iOS)
            VStack {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(selection.channel.name)
                            .font(.headline)
                        if let program = selection.program {
                            Text(program.title)
                                .font(.subheadline)
                                .foregroundStyle(.white.opacity(0.7))
                        }
                    }
                    Spacer()
                    if let player, let timeline = archiveTimeline {
                        VStack(spacing: 12) {
                            ArchiveSeekBar(
                                player: player, timeline: timeline, isScrubbing: $isScrubbing,
                                seek: seekArchive
                            )
                            ArchivePlayPauseButton(player: player)
                        }
                        .padding()
                        .background(.black.opacity(0.7))
                        .disabled(isPreparingArchive)
                    }
                    if let startOver = model.startOverForSelection {
                        Button(action: startOver) {
                            Image(systemName: "backward.end.fill")
                                .frame(width: 36, height: 36)
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel("Start over")
                    }
                    if selection.isCatchup {
                        Button("Live") { model.play(selection.channel) }
                            .font(.caption.weight(.bold))
                            .buttonStyle(.bordered)
                            .tint(.white)
                            .accessibilityLabel("Return to live")
                    }
                    if let quality {
                        Text(quality)
                            .font(.caption.weight(.semibold))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(.white.opacity(0.18), in: Capsule())
                    }
                    if let player {
                        AirPlayButton(player: player)
                            .frame(width: 36, height: 36)
                            .accessibilityLabel("AirPlay")
                    }
                }
                .padding()
                .background(
                    LinearGradient(
                        colors: [.black.opacity(0.75), .clear],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                Spacer()
            }
            .foregroundStyle(.white)
            #endif
        }
        #if os(macOS) || os(tvOS)
        #if os(macOS)
        .overlay(alignment: .bottom) {
            if let player {
                MacControlBar(
                    player: player, volume: $model.playbackVolume, airPlayActive: airPlayActive,
                    isCatchup: selection.isCatchup, startOver: model.startOverForSelection,
                    goLive: selection.isCatchup ? { model.play(selection.channel) } : nil,
                    timeline: archiveTimeline, isScrubbing: $isScrubbing, seek: seekArchive
                )
                .disabled(isPreparingArchive)
            }
        }
        .overlay(alignment: .top) {
            if airPlayActive, let airPlayHost {
                AirPlayHostWarning(host: airPlayHost)
                    .padding(12)
            }
        }
        #endif
        .overlay(alignment: .topLeading) {
            if let quality {
                Text(quality)
                    .font(.caption.weight(.semibold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .foregroundStyle(.white)
                    .background(.black.opacity(0.65), in: Capsule())
                    #if os(tvOS)
                    .padding(model.isPlayerExpanded ? 48 : 18)
                    #else
                    .padding(12)
                    #endif
                    .allowsHitTesting(false)
            }
        }
        .onChange(of: model.playbackVolume) { _, volume in
            player?.volume = Float(volume)
        }
        #endif
        #if os(tvOS)
        .overlay(alignment: .bottom) {
            if let player {
                TVControlBar(
                    player: player,
                    selection: selection,
                    expanded: model.isPlayerExpanded,
                    lastActivity: tvActivity,
                    timeline: archiveTimeline,
                    archiveSeekPosition: tvSeekPosition,
                    liveSeekPosition: tvLiveSeekPosition,
                    liveBufferDuration: liveBufferDuration,
                    liveProgram: liveProgram
                )
            }
        }
        .onChange(of: model.playPauseRequest) { _, _ in
            guard !isPreparingArchive else { return }
            if let position = tvSeekPosition {
                tvSeekPosition = nil
                seekArchive(position, true)
            } else if tvLiveSeekPosition != nil {
                let wasPlaying = tvLiveResumeAfterSeek
                seekTask?.cancel()
                player?.currentItem?.cancelPendingSeeks()
                tvLiveSeekPosition = nil
                tvLiveResumeAfterSeek = false
                if wasPlaying { player?.pause() } else { resumeTVPlayback() }
            } else if player?.timeControlStatus == .paused {
                resumeTVPlayback()
            } else {
                seekTask?.cancel()
                player?.currentItem?.cancelPendingSeeks()
                tvLiveSeekPosition = nil
                tvLiveResumeAfterSeek = false
                player?.pause()
            }
            tvActivity = Date()
        }
        .onChange(of: model.playerActivity) { _, _ in
            tvActivity = Date()
        }
        .onChange(of: model.seekBackwardRequest) { _, _ in handleTVSeek(direction: -1) }
        .onChange(of: model.seekForwardRequest) { _, _ in handleTVSeek(direction: 1) }
        .onChange(of: model.isPlayerExpanded) { _, expanded in
            if expanded { tvActivity = Date() }
            else {
                if tvSeekPosition != nil && tvResumeAfterCancel { player?.play() }
                tvSeekPosition = nil
                tvLiveSeekPosition = nil
                tvLiveResumeAfterSeek = false
            }
        }
        #endif
        .task {
            await runPlayback()
        }
        .onDisappear {
            seekTask?.cancel()
            player?.pause()
        }
        .alert("Unable to Seek", isPresented: Binding(
            get: { seekError != nil }, set: { if !$0 { seekError = nil } }
        )) {
            Button("OK") { seekError = nil }
        } message: {
            Text(seekError ?? "")
        }
    }

    #if os(tvOS)
    private func handleTVSeek(direction: Double) {
        if selection.isCatchup {
            previewTVSeek(by: direction * 10)
        } else {
            seekLive(by: direction * 15)
        }
    }

    private func previewTVSeek(by seconds: Double) {
        guard model.isPlayerExpanded, !isPreparingArchive,
              let player, let timeline = archiveTimeline else { return }
        if tvSeekPosition == nil { tvResumeAfterCancel = player.timeControlStatus != .paused }
        let position = tvSeekPosition ?? timeline.elapsed(mediaTime: player.currentTime().seconds)
        tvSeekPosition = min(max(position + seconds, 0), timeline.duration)
        player.pause()
        tvActivity = Date()
    }

    private func seekLive(by seconds: Double) {
        guard model.isPlayerExpanded, !isPreparingArchive, let player else { return }
        let base = tvLiveSeekPosition ?? player.currentTime().seconds
        guard let timeline = makeLiveTimeline(
            player: player, position: base, maximumDuration: liveBufferDuration
        ) else { return }
        if tvLiveSeekPosition == nil {
            tvLiveResumeAfterSeek = player.timeControlStatus != .paused
        }
        seekLive(
            player, to: timeline.seekTarget(offsetBy: seconds),
            resume: tvLiveResumeAfterSeek
        )
    }

    private func resumeTVPlayback() {
        guard let player else { return }
        guard !selection.isCatchup,
              let timeline = makeLiveTimeline(
                player: player, maximumDuration: liveBufferDuration
              ),
              player.currentTime().seconds < timeline.start else {
            player.play()
            return
        }
        seekLive(player, to: timeline.seekTarget(offsetBy: 0), resume: true)
    }

    private func seekLive(_ player: AVPlayer, to position: Double, resume: Bool) {
        seekTask?.cancel()
        player.currentItem?.cancelPendingSeeks()
        tvLiveSeekPosition = position
        tvLiveResumeAfterSeek = resume
        tvActivity = Date()
        seekTask = Task { @MainActor in
            let tolerance = CMTime(seconds: 1, preferredTimescale: 600)
            let sought = await player.seek(
                to: CMTime(seconds: position, preferredTimescale: 600),
                toleranceBefore: tolerance,
                toleranceAfter: tolerance
            )
            guard !Task.isCancelled else { return }
            guard sought else {
                if tvLiveSeekPosition == position {
                    tvLiveSeekPosition = nil
                    tvLiveResumeAfterSeek = false
                }
                seekError = "The retained live position could not be loaded."
                return
            }
            if resume { player.play() } else { player.pause() }
            if tvLiveSeekPosition == position {
                tvLiveSeekPosition = nil
                tvLiveResumeAfterSeek = false
            }
        }
    }
    #endif

    @MainActor
    private func seekArchive(_ elapsed: Double, _ resume: Bool) {
        guard !isPreparingArchive, let player, let timeline = archiveTimeline, elapsed.isFinite else { return }
        seekTask?.cancel()
        let ranges = seekableRanges(in: player.currentItem)
        let position = timeline.localPosition(elapsed: elapsed, seekable: ranges)
        player.pause()
        seekTask = Task { @MainActor in
            guard !Task.isCancelled else { return }
            if let position {
                let sought = await player.seek(
                    to: CMTime(seconds: position, preferredTimescale: 600),
                    toleranceBefore: .zero, toleranceAfter: .zero
                )
                guard !Task.isCancelled else { return }
                if sought {
                    if resume { player.play() }
                    return
                }
            }
            do {
                try model.seekCatchup(selection, to: timeline.timestamp(elapsed: elapsed), resume: resume)
            } catch {
                logger.error("Archive seek failed: \(error.localizedDescription, privacy: .public)")
                seekError = error.localizedDescription
            }
        }
    }

    @MainActor
    private func runPlayback() async {
        var activeSessionID: String?
        defer {
            player?.pause()
            player = nil
            quality = nil
            airPlayActive = false
            isPreparingArchive = false
            liveBufferDuration = 0
            if let sessionID = activeSessionID {
                Task { await model.stopPlayback(sessionID: sessionID) }
            }
        }
        do {
            #if os(iOS)
            try configureAudioSession()
            #endif
            var nextLiveGuideRefresh = Date.distantPast
            while !Task.isCancelled {
                let bandwidthSaver = model.bandwidthSaver
                var configuration = try await model.playerConfiguration(for: selection)
                activeSessionID = configuration.transcodeSessionID
                liveBufferDuration = selection.isCatchup ? 0 : configuration.liveBufferDuration
                try Task.checkCancellation()
                var options: [String: Any] = [:]
                if let cookie = configuration.cookieHeader {
                    options["AVURLAssetHTTPHeaderFieldsKey"] = ["Cookie": cookie]
                }
                let asset = AVURLAsset(url: configuration.url, options: options)
                guard try await asset.load(.isPlayable) else {
                    throw APIError.server("This channel's stream is not compatible with AVPlayer.")
                }
                try Task.checkCancellation()
                var item = AVPlayerItem(asset: asset)
                item.preferredForwardBufferDuration = 12
                if !selection.isCatchup {
                    item.automaticallyPreservesTimeOffsetFromLive = true
                }
                var currentPlayer = AVPlayer(playerItem: item)
                currentPlayer.isMuted = false
                #if os(iOS)
                currentPlayer.usesExternalPlaybackWhileExternalScreenIsActive = true
                #endif
                airPlayHost = configuration.url.host.flatMap { isLoopback($0) ? $0 : nil }
                #if os(macOS) || os(tvOS)
                currentPlayer.volume = Float(model.playbackVolume)
                #else
                currentPlayer.volume = 1
                #endif
                isPreparingArchive = selection.isCatchup
                player = currentPlayer
                if let requested = selection.catchupStart {
                    archiveStreamStart = configuration.archiveStart ?? floor(requested / 60) * 60
                    let offset = configuration.archiveSeek ?? max(0, requested - (archiveStreamStart ?? requested))
                    try await prepareArchivePosition(
                        player: currentPlayer, offset: offset, sessionID: activeSessionID
                    )
                }
                isPreparingArchive = false
                if !selection.startPaused { currentPlayer.play() }
                var sampler = PlaybackHealthSampler()
                var shouldRetune = false
                var deferredSaverDecision: Bool?
                while !Task.isCancelled {
                    try await Task.sleep(for: .seconds(2))
                    quality = qualityLabel(for: item.presentationSize)
                    if item.status == .failed {
                        throw item.error ?? APIError.server("The stream could not be played.")
                    }
                    if model.isPlayerExpanded, !selection.isCatchup, liveProgram == nil,
                       Date() >= nextLiveGuideRefresh {
                        nextLiveGuideRefresh = Date().addingTimeInterval(30)
                        await model.refreshGuideOnReturn()
                    }
                    if selection.isCatchup, let sessionID = activeSessionID {
                        do {
                            try await model.keepArchiveAlive(sessionID: sessionID)
                        } catch {
                            if Task.isCancelled { throw CancellationError() }
                            logger.warning("Archive heartbeat failed: \(error.localizedDescription, privacy: .public)")
                        }
                    }
                    #if os(iOS) || os(macOS)
                    airPlayActive = currentPlayer.isExternalPlaybackActive
                    // The TV fetches segments itself, which keeps the session alive. Swapping
                    // players for a quality change would drop the AirPlay route.
                    if airPlayActive { continue }
                    #endif
                    // Archived programs play as VOD, which takes no live quality feedback.
                    guard let sessionID = activeSessionID, !selection.isCatchup else { continue }
                    // Pauses aren't stalls. Reporting healthy samples also resets the
                    // server's consecutive-poor-playback window and keeps it alive.
                    let health = sampler.sample(player: currentPlayer, item: item)
                    do {
                        let feedback = try await model.reportPlaybackHealth(
                            sessionID: sessionID, health: health
                        )
                        try Task.checkCancellation()
                        if let playlist = feedback.playlist,
                           let url = URL(string: playlist, relativeTo: configuration.url)?.absoluteURL,
                           url != configuration.url {
                            // Prepare locally while the current rendition keeps playing.
                            let replacement = try await prepareQualityPlayer(
                                url: url, options: options, currentItem: item
                            )
                            try Task.checkCancellation()
                            replacement.volume = currentPlayer.volume
                            replacement.isMuted = currentPlayer.isMuted
                            let wasPaused = currentPlayer.timeControlStatus == .paused
                            currentPlayer.pause()
                            currentPlayer = replacement
                            item = replacement.currentItem!
                            player = replacement
                            if !wasPaused { replacement.play() }
                            configuration = PlaybackConfiguration(
                                url: url, cookieHeader: configuration.cookieHeader,
                                transcodeSessionID: sessionID,
                                liveBufferDuration: configuration.liveBufferDuration
                            )
                            sampler = PlaybackHealthSampler()
                            continue
                        }
                        if feedback.playlist == nil && feedback.bandwidthSaver != bandwidthSaver {
                            if configuration.liveBufferDuration > 0 {
                                if deferredSaverDecision != feedback.bandwidthSaver {
                                    logger.info(
                                        "Keeping the current live session to preserve its retained playback window"
                                    )
                                    deferredSaverDecision = feedback.bandwidthSaver
                                }
                            } else {
                                shouldRetune = true
                                break
                            }
                        }
                    } catch {
                        if Task.isCancelled { throw CancellationError() }
                        // Older servers or transient telemetry failures must not stop playback.
                        logger.debug("Playback health unavailable: \(error.localizedDescription, privacy: .public)")
                    }
                }
                if shouldRetune {
                    logger.info("Switching playback quality")
                    currentPlayer.pause()
                    player = nil
                    quality = nil
                    if let sessionID = activeSessionID {
                        await model.stopPlayback(sessionID: sessionID)
                        activeSessionID = nil
                    }
                }
            }
        } catch {
            if !Task.isCancelled {
                logger.error("Playback failed: \(error.localizedDescription, privacy: .public)")
                errorMessage = error.localizedDescription
            }
        }
    }

    @MainActor
    private func prepareArchivePosition(player: AVPlayer, offset: Double, sessionID: String?) async throws {
        let deadline = Date().addingTimeInterval(40)
        var nextHeartbeat = Date()
        while Date() < deadline {
            try Task.checkCancellation()
            guard let item = player.currentItem, item.status != .failed else {
                throw APIError.server("The archive could not be loaded.")
            }
            if let sessionID, Date() >= nextHeartbeat {
                try await model.keepArchiveAlive(sessionID: sessionID)
                nextHeartbeat = Date().addingTimeInterval(2)
            }
            if item.status == .readyToPlay && item.seekableTimeRanges.contains(where: {
                let range = $0.timeRangeValue
                return range.start.seconds <= offset && CMTimeRangeGetEnd(range).seconds > offset
            }) {
                let sought = await player.seek(
                    to: CMTime(seconds: offset, preferredTimescale: 600),
                    toleranceBefore: .zero, toleranceAfter: .zero
                )
                guard sought else { throw APIError.server("The archive position could not be loaded.") }
                return
            }
            try await Task.sleep(for: .milliseconds(200))
        }
        throw APIError.server("Timed out waiting for the selected archive position.")
    }

    @MainActor
    private func prepareQualityPlayer(
        url: URL, options: [String: Any], currentItem: AVPlayerItem
    ) async throws -> AVPlayer {
        let item = AVPlayerItem(asset: AVURLAsset(url: url, options: options))
        item.preferredForwardBufferDuration = 12
        item.automaticallyPreservesTimeOffsetFromLive = true
        let candidate = AVPlayer(playerItem: item)
        let deadline = Date().addingTimeInterval(8)
        while item.status == .unknown && Date() < deadline {
            try await Task.sleep(for: .milliseconds(100))
        }
        guard item.status == .readyToPlay else {
            throw APIError.server("The new quality is not ready yet.")
        }
        // Both playlists expose dates derived from the same provider timestamps.
        // Refuse an upgrade without a shared timeline rather than jumping live.
        guard let date = currentItem.currentDate() else {
            throw APIError.server("Waiting for the shared playback timeline.")
        }
        let sought = await withCheckedContinuation { continuation in
            item.seek(to: date) { success in
                continuation.resume(returning: success)
            }
        }
        guard sought else {
            throw APIError.server("The new quality has not caught up yet.")
        }
        return candidate
    }

    #if os(iOS)
    private func configureAudioSession() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .moviePlayback)
        try session.setActive(true)
        let outputs = session.currentRoute.outputs
            .map { $0.portType.rawValue }
            .joined(separator: ", ")
        logger.info(
            "Audio session active: route=\(outputs.isEmpty ? "none" : outputs, privacy: .public) outputVolume=\(session.outputVolume)"
        )
    }
    #endif
}

#if os(macOS)
private struct PlayerController: NSViewRepresentable {
    let player: AVPlayer
    let isCatchup: Bool

    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.controlsStyle = .none
        view.videoGravity = .resizeAspect
        view.player = player
        return view
    }

    func updateNSView(_ view: AVPlayerView, context: Context) {
        if view.player !== player {
            view.player = player
        }
    }
}

/// A slim bar of our own: AVKit's inline controls crashed with neTV's live streams.
/// It appears on hover and stays visible while paused.
private struct MacControlBar: View {
    let player: AVPlayer
    @Binding var volume: Double
    let airPlayActive: Bool
    let isCatchup: Bool
    let startOver: (() -> Void)?
    let goLive: (() -> Void)?
    let timeline: ArchiveTimeline?
    @Binding var isScrubbing: Bool
    let seek: (Double, Bool) -> Void

    @State private var isPlaying = true
    @State private var isHovering = false
    @State private var lastActivity = Date.distantPast

    var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { context in
            let visible = isScrubbing || !isPlaying || (isHovering && context.date.timeIntervalSince(lastActivity) < 3)
            VStack(spacing: 8) {
                if let timeline {
                    ArchiveSeekBar(player: player, timeline: timeline, isScrubbing: $isScrubbing, seek: seek)
                        .padding(.horizontal, 12)
                        .padding(.top, 12)
                        .background(.black.opacity(0.7))
                }
                bar
            }
                .opacity(visible ? 1 : 0)
                .animation(.easeInOut(duration: 0.2), value: visible)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
        .contentShape(Rectangle())
        .onContinuousHover { phase in
            if case .active = phase {
                isHovering = true
                lastActivity = Date()
            } else {
                isHovering = false
            }
        }
        .onReceive(player.publisher(for: \.timeControlStatus)) { status in
            isPlaying = status != .paused
        }
    }

    private var bar: some View {
        HStack(spacing: 12) {
            Button {
                if player.timeControlStatus == .paused { player.play() } else { player.pause() }
                lastActivity = Date()
            } label: {
                Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                    .font(.system(size: 13, weight: .semibold))
                    .frame(width: 20, height: 20)
            }
            .buttonStyle(.plain)
            .help(isPlaying ? "Pause" : "Play")
            .accessibilityLabel(isPlaying ? "Pause" : "Play")

            if let startOver {
                Button(action: startOver) {
                    Image(systemName: "backward.end.fill")
                        .font(.system(size: 12, weight: .semibold))
                        .frame(width: 20, height: 20)
                }
                .buttonStyle(.plain)
                .help("Start over")
                .accessibilityLabel("Start over")
            }

            if isCatchup {
                CatchupBadge()
                if let goLive {
                    Button("Go Live", action: goLive)
                        .font(.caption2.weight(.bold))
                        .buttonStyle(.plain)
                        .help("Return to the live broadcast")
                }
            } else {
                HStack(spacing: 4) {
                    Circle().fill(.red).frame(width: 6, height: 6)
                    Text("LIVE").font(.caption2.weight(.bold))
                }
                .accessibilityElement(children: .combine)
            }

            Spacer(minLength: 8)

            Image(systemName: volume == 0 ? "speaker.slash.fill" : "speaker.wave.2.fill")
                .font(.system(size: 12))
                .frame(width: 16)
                .accessibilityHidden(true)
            Slider(value: $volume, in: 0...1) { editing in
                if editing { lastActivity = Date() }
            }
            .labelsHidden()
            .controlSize(.mini)
            .tint(.white)
            .frame(width: 70)
            .accessibilityLabel("Volume")
            .accessibilityValue("\(Int(volume * 100)) percent")

            AirPlayButton(player: player, active: airPlayActive)
                .frame(width: 22, height: 22)
                .help("AirPlay")
                .accessibilityLabel("AirPlay")
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 10)
        .frame(height: 30)
        .background(
            LinearGradient(colors: [.clear, .black.opacity(0.8)], startPoint: .top, endPoint: .bottom)
        )
    }
}

/// macOS's route picker always draws the AirPlay audio glyph, so its button is made
/// transparent and the video glyph is drawn on top without taking the click.
private struct AirPlayButton: View {
    let player: AVPlayer
    let active: Bool

    var body: some View {
        ZStack {
            RoutePicker(player: player)
            Image(systemName: "airplayvideo")
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(active ? Color.blue : Color.white)
                .allowsHitTesting(false)
        }
    }
}

/// Routes this player's video; AVRoutePickerView.player is macOS-only.
private struct RoutePicker: NSViewRepresentable {
    let player: AVPlayer

    func makeNSView(context: Context) -> AVRoutePickerView {
        let view = AVRoutePickerView()
        view.isRoutePickerButtonBordered = false
        for state in [AVRoutePickerView.ButtonState.normal, .normalHighlighted, .active, .activeHighlighted] {
            view.setRoutePickerButtonColor(.clear, for: state)
        }
        view.player = player
        return view
    }

    func updateNSView(_ view: AVRoutePickerView, context: Context) {
        if view.player !== player {
            view.player = player
        }
    }
}

private struct AirPlayHostWarning: View {
    let host: String

    var body: some View {
        Label(
            "The TV can't reach \(host). Sign in with this Mac's LAN address, such as http://192.168.1.10:8000.",
            systemImage: "exclamationmark.triangle.fill"
        )
        .font(.caption)
        .foregroundStyle(.white)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.black.opacity(0.75), in: RoundedRectangle(cornerRadius: 10))
        .allowsHitTesting(false)
    }
}
#elseif os(iOS)
private struct AirPlayButton: UIViewRepresentable {
    let player: AVPlayer

    func makeUIView(context: Context) -> AVRoutePickerView {
        let view = AVRoutePickerView()
        view.prioritizesVideoDevices = true
        view.tintColor = .white
        view.activeTintColor = .systemBlue
        return view
    }

    func updateUIView(_ view: AVRoutePickerView, context: Context) {}
}

private struct PlayerController: UIViewControllerRepresentable {
    let player: AVPlayer
    let isCatchup: Bool

    func makeUIViewController(context: Context) -> AVPlayerViewController {
        let controller = AVPlayerViewController()
        controller.player = player
        controller.showsPlaybackControls = !isCatchup
        controller.allowsPictureInPicturePlayback = true
        controller.canStartPictureInPictureAutomaticallyFromInline = true
        controller.updatesNowPlayingInfoCenter = true
        return controller
    }

    func updateUIViewController(_ controller: AVPlayerViewController, context: Context) {
        controller.player = player
        controller.showsPlaybackControls = !isCatchup
    }
}
#else
private struct PlayerController: UIViewControllerRepresentable {
    let player: AVPlayer
    let isCatchup: Bool

    func makeUIViewController(context: Context) -> AVPlayerViewController {
        let controller = AVPlayerViewController()
        controller.showsPlaybackControls = false
        controller.videoGravity = .resizeAspect
        controller.player = player
        return controller
    }

    func updateUIViewController(_ controller: AVPlayerViewController, context: Context) {
        if controller.player !== player {
            controller.player = player
        }
    }
}

/// Status-only bar: the Siri Remote drives playback, so nothing here takes focus.
/// It appears on remote activity and stays visible while paused.
private struct TVControlBar: View {
    let player: AVPlayer
    let selection: PlayerSelection
    let expanded: Bool
    let lastActivity: Date
    let timeline: ArchiveTimeline?
    let archiveSeekPosition: Double?
    let liveSeekPosition: Double?
    let liveBufferDuration: Double
    let liveProgram: Program?

    @State private var isPlaying = true

    var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { context in
            let liveTimeline = selection.isCatchup ? nil : makeLiveTimeline(
                player: player,
                position: liveSeekPosition,
                maximumDuration: liveBufferDuration
            )
            let programTimeline = liveTimeline.flatMap {
                makeLiveProgramTimeline(
                    player: player,
                    program: liveProgram,
                    liveTimeline: $0,
                    fallbackLiveDate: context.date
                )
            }
            let visible = archiveSeekPosition != nil || liveSeekPosition != nil
                || !isPlaying || context.date.timeIntervalSince(lastActivity) < 4
            VStack(spacing: 12) {
                if let timeline {
                    let elapsed = archiveSeekPosition
                        ?? timeline.elapsed(mediaTime: player.currentTime().seconds)
                    VStack(spacing: 8) {
                        ProgressView(value: elapsed, total: timeline.duration)
                            .tint(.white)
                        HStack {
                            Text(archiveTime(elapsed))
                            Spacer()
                            Text(
                                archiveSeekPosition == nil
                                    ? "Left/Right to seek" : "Select or Play to seek"
                            )
                            Spacer()
                            Text("-" + archiveTime(timeline.duration - elapsed))
                        }
                        .font(expanded ? .caption : .caption2)
                        .monospacedDigit()
                    }
                    .foregroundStyle(.white)
                    .padding(.horizontal, expanded ? 60 : 16)
                } else if let liveTimeline {
                    VStack(spacing: 8) {
                        if let programTimeline {
                            LiveProgramProgressBar(timeline: programTimeline)
                            HStack {
                                Text(programTime(programTimeline.start))
                                Spacer()
                                Text(
                                    "\(programTime(programTimeline.playback))  •  "
                                        + liveStatus(liveTimeline)
                                )
                                Spacer()
                                Text(programTime(programTimeline.end))
                            }
                            .font(expanded ? .caption : .caption2)
                            .monospacedDigit()
                        } else {
                            ProgressView(value: liveTimeline.elapsed, total: liveTimeline.duration)
                                .tint(.white)
                            HStack {
                                Text("-" + archiveTime(liveTimeline.duration))
                                Spacer()
                                Text(liveStatus(liveTimeline))
                                Spacer()
                                HStack(spacing: 5) {
                                    Circle().fill(.red).frame(width: 6, height: 6)
                                    Text("LIVE")
                                }
                            }
                            .font(expanded ? .caption : .caption2)
                            .monospacedDigit()
                        }
                    }
                    .foregroundStyle(.white)
                    .padding(.horizontal, expanded ? 60 : 16)
                }
                bar(liveTimeline: liveTimeline)
            }
                .opacity(visible ? 1 : 0)
                .animation(.easeInOut(duration: 0.3), value: visible)
        }
        .allowsHitTesting(false)
        .onReceive(player.publisher(for: \.timeControlStatus)) { status in
            isPlaying = status != .paused
        }
    }

    private func liveStatus(_ timeline: LiveTimeline) -> String {
        let position = timeline.isAtLiveEdge
            ? "At live edge" : "\(archiveTime(timeline.behindLive)) behind"
        return expanded ? "\(position)  •  Left/Right 15s" : position
    }

    private func bar(liveTimeline: LiveTimeline?) -> some View {
        HStack(spacing: expanded ? 24 : 12) {
            Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                .font(.system(size: expanded ? 30 : 18, weight: .semibold))
                .frame(width: expanded ? 36 : 22)
                .accessibilityLabel(isPlaying ? "Playing" : "Paused")

            if selection.isCatchup {
                CatchupBadge()
            } else if let liveTimeline, !liveTimeline.isAtLiveEdge {
                HStack(spacing: 6) {
                    Image(systemName: "clock.arrow.circlepath")
                    Text("-\(archiveTime(liveTimeline.behindLive)) LIVE")
                        .font((expanded ? Font.callout : .caption2).weight(.bold))
                }
                .accessibilityElement(children: .combine)
                .accessibilityLabel("\(archiveTime(liveTimeline.behindLive)) behind live")
            } else {
                HStack(spacing: 6) {
                    Circle().fill(.red).frame(width: expanded ? 10 : 7, height: expanded ? 10 : 7)
                    Text("LIVE").font((expanded ? Font.callout : .caption2).weight(.bold))
                }
                .accessibilityElement(children: .combine)
            }

            if expanded {
                VStack(alignment: .leading, spacing: 2) {
                    Text(selection.channel.name)
                        .font(.headline)
                    if let program = selection.isCatchup ? selection.program : liveProgram {
                        Text("\(program.title)  \(program.start) – \(program.end)")
                            .font(.subheadline)
                            .foregroundStyle(.white.opacity(0.7))
                    }
                }
                .lineLimit(1)
            }

            Spacer(minLength: 0)
        }
        .foregroundStyle(.white)
        .padding(.horizontal, expanded ? 60 : 16)
        .padding(.top, expanded ? 40 : 16)
        .padding(.bottom, expanded ? 50 : 10)
        .background(
            LinearGradient(colors: [.clear, .black.opacity(0.85)], startPoint: .top, endPoint: .bottom)
        )
    }
}

private struct LiveProgramProgressBar: View {
    let timeline: LiveProgramTimeline

    var body: some View {
        GeometryReader { proxy in
            let width = proxy.size.width
            let playbackWidth = width * CGFloat(timeline.playbackProgress)
            let liveWidth = width * CGFloat(timeline.liveProgress)
            let markerRadius: CGFloat = 7
            let markerMaximum = max(markerRadius, width - markerRadius)
            let playbackX = min(max(playbackWidth, markerRadius), markerMaximum)
            let liveX = min(max(liveWidth, markerRadius), markerMaximum)
            let labelMargin: CGFloat = 18
            let labelMaximum = max(labelMargin, width - labelMargin)
            let liveLabelX = min(max(liveWidth, labelMargin), labelMaximum)

            ZStack(alignment: .leading) {
                ZStack(alignment: .leading) {
                    Capsule().fill(.white.opacity(0.16))
                    Rectangle()
                        .fill(.white.opacity(0.38))
                        .frame(width: liveWidth)
                    Rectangle()
                        .fill(.white.opacity(0.9))
                        .frame(width: playbackWidth)
                }
                .frame(width: width, height: 8)
                .clipShape(Capsule())
                .position(x: width / 2, y: 20)

                Circle()
                    .fill(.black.opacity(0.5))
                    .overlay(Circle().stroke(.white, lineWidth: 3))
                    .frame(width: markerRadius * 2, height: markerRadius * 2)
                    .position(x: playbackX, y: 20)

                Capsule()
                    .fill(.red)
                    .frame(width: 3, height: 18)
                    .position(x: liveX, y: 20)

                Text("LIVE")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(.red)
                    .position(x: liveLabelX, y: 5)
            }
        }
        .frame(height: 30)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(
            "Program \(programTime(timeline.start)) to \(programTime(timeline.end)), "
                + "playback at \(programTime(timeline.playback)), "
                + "live at \(programTime(timeline.live))"
        )
    }
}

#endif

private func archiveTime(_ seconds: Double) -> String {
    let value = Int(max(0, seconds.isFinite ? seconds : 0))
    return value >= 3600
        ? String(format: "%d:%02d:%02d", value / 3600, value / 60 % 60, value % 60)
        : String(format: "%d:%02d", value / 60, value % 60)
}

private func programTime(_ timestamp: Double) -> String {
    Date(timeIntervalSince1970: timestamp).formatted(date: .omitted, time: .shortened)
}

private func seekableRanges(in item: AVPlayerItem?) -> [ClosedRange<Double>] {
    item?.seekableTimeRanges.compactMap { value in
        let range = value.timeRangeValue
        let start = range.start.seconds
        let end = CMTimeRangeGetEnd(range).seconds
        return start.isFinite && end.isFinite && end > start ? start...end : nil
    } ?? []
}

private func makeLiveTimeline(
    player: AVPlayer, position: Double? = nil, maximumDuration: Double
) -> LiveTimeline? {
    LiveTimeline(
        seekable: seekableRanges(in: player.currentItem),
        position: position ?? player.currentTime().seconds,
        maximumDuration: maximumDuration
    )
}

private func makeLiveProgramTimeline(
    player: AVPlayer,
    program: Program?,
    liveTimeline: LiveTimeline,
    fallbackLiveDate: Date
) -> LiveProgramTimeline? {
    let currentPosition = player.currentTime().seconds
    let playbackTimestamp: Double
    if currentPosition.isFinite, let currentDate = player.currentItem?.currentDate() {
        playbackTimestamp = currentDate.timeIntervalSince1970
            + liveTimeline.position - currentPosition
    } else {
        playbackTimestamp = fallbackLiveDate.timeIntervalSince1970 - liveTimeline.behindLive
    }
    return LiveProgramTimeline(
        program: program,
        playbackTimestamp: playbackTimestamp,
        liveTimestamp: playbackTimestamp + liveTimeline.behindLive
    )
}

#if !os(tvOS)
private struct ArchiveSeekBar: View {
    let player: AVPlayer
    let timeline: ArchiveTimeline
    @Binding var isScrubbing: Bool
    let seek: (Double, Bool) -> Void
    @State private var draft = 0.0
    @State private var resumeAfterSeek = true

    var body: some View {
        TimelineView(.periodic(from: .now, by: 0.5)) { _ in
            let current = timeline.elapsed(mediaTime: player.currentTime().seconds)
            let displayed = isScrubbing ? draft : current
            VStack(spacing: 4) {
                HStack(spacing: 12) {
                    Button {
                        seek(current - 10, player.timeControlStatus != .paused)
                    } label: {
                        Image(systemName: "gobackward.10")
                    }
                    .accessibilityLabel("Back 10 seconds")
                    Slider(value: Binding(
                        get: { isScrubbing ? draft : current },
                        set: { draft = $0 }
                    ), in: 0...timeline.duration) { editing in
                        if editing {
                            draft = current
                            resumeAfterSeek = player.timeControlStatus != .paused
                            isScrubbing = true
                            player.pause()
                        } else {
                            isScrubbing = false
                            seek(draft, resumeAfterSeek)
                        }
                    }
                    .tint(.white)
                    .accessibilityLabel("Archive playback position")
                    .accessibilityValue(archiveTime(displayed))
                    Button {
                        seek(current + 10, player.timeControlStatus != .paused)
                    } label: {
                        Image(systemName: "goforward.10")
                    }
                    .accessibilityLabel("Forward 10 seconds")
                }
                .buttonStyle(.plain)
                HStack {
                    Text(archiveTime(displayed))
                    Spacer()
                    Text("-" + archiveTime(timeline.duration - displayed))
                }
                .font(.caption.monospacedDigit())
            }
            .foregroundStyle(.white)
        }
    }
}

private struct ArchivePlayPauseButton: View {
    let player: AVPlayer
    @State private var paused = false

    var body: some View {
        Button {
            paused ? player.play() : player.pause()
        } label: {
            Image(systemName: paused ? "play.fill" : "pause.fill")
                .frame(width: 44, height: 36)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(paused ? "Play" : "Pause")
        .onReceive(player.publisher(for: \.timeControlStatus)) { paused = $0 == .paused }
    }
}
#endif

private func isLoopback(_ host: String) -> Bool {
    host == "localhost" || host == "::1" || host.hasPrefix("127.")
}

/// Classify on the larger of the frame height and the height a 16:9 frame of this width
/// would have: a letterboxed 1920x800 frame is a 1080p stream, not a 720p one.
private func qualityLabel(for size: CGSize) -> String? {
    guard size.width > 0, size.height > 0 else { return nil }
    switch max(size.height, size.width * 9 / 16) {
    case 2000...: return "4K"
    case 1300...: return "1440p"
    case 900...: return "1080p"
    case 650...: return "720p"
    case 520...: return "576p"
    case 400...: return "480p"
    default: return "SD"
    }
}

/// Use recent transfer deltas; a cumulative bitrate can hide a Wi-Fi-to-cellular slowdown.
private struct PlaybackHealthSampler {
    private var lastBytes: Int64 = 0
    private var lastTransferDuration: Double = 0
    private var lastEventStart: Date?

    mutating func sample(player: AVPlayer, item: AVPlayerItem) -> PlaybackHealth {
        let position = item.currentTime().seconds
        let buffer = item.loadedTimeRanges.reduce(0.0) { result, value in
            let range = value.timeRangeValue
            let start = range.start.seconds
            let end = CMTimeRangeGetEnd(range).seconds
            return start <= position && position <= end ? max(result, end - position) : result
        }
        var observed = 0.0
        var required = 0.0
        if let event = item.accessLog()?.events.last {
            if event.playbackStartDate != lastEventStart {
                lastBytes = 0
                lastTransferDuration = 0
                lastEventStart = event.playbackStartDate
            }
            let bytes = event.numberOfBytesTransferred - lastBytes
            let duration = event.transferDuration - lastTransferDuration
            if bytes > 0 && duration > 0 { observed = Double(bytes) * 8 / duration }
            lastBytes = event.numberOfBytesTransferred
            lastTransferDuration = event.transferDuration
            required = event.indicatedBitrate
            if required <= 0 {
                required = max(0, event.averageVideoBitrate) + max(0, event.averageAudioBitrate)
            }
        }
        let paused = player.timeControlStatus == .paused
        return PlaybackHealth(
            bufferSeconds: buffer.isFinite ? max(0, buffer) : 0,
            waiting: player.timeControlStatus == .waitingToPlayAtSpecifiedRate,
            observedBitrate: !paused && observed.isFinite ? max(0, observed) : 0,
            requiredBitrate: !paused && required.isFinite ? max(0, required) : 0
        )
    }
}
