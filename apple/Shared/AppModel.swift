import Combine
import Foundation
import OSLog

@MainActor
final class AppModel: ObservableObject {
    @Published var isAuthenticated = false
    @Published var isCheckingSession = true
    @Published var isLoading = false
    @Published var channels: [ChannelRow] = []
    @Published var guideCategories: [GuideCategory] = []
    @Published var guideWindowStart = Date(
        timeIntervalSince1970: floor(Date().timeIntervalSince1970 / 3600) * 3600
    )
    @Published private(set) var guideOffset = 0
    @Published private(set) var requestedGuideOffset = 0
    private var guideLoadGeneration = 0
    @Published var query = ""
    @Published var errorMessage: String?
    @Published var selection: PlayerSelection?
    @Published var isPlayerExpanded = false
    #if os(macOS) || os(tvOS)
    @Published var playbackVolume: Double = 1
    #endif
    #if os(tvOS)
    @Published var playPauseRequest: UUID?
    @Published var playerActivity: UUID?
    @Published var seekBackwardRequest: UUID?
    @Published var seekForwardRequest: UUID?
    #endif

    // Carry the server's current quality decision across channel changes.
    private(set) var bandwidthSaver = false

    @Published var server: String {
        didSet { UserDefaults.standard.set(server, forKey: "server") }
    }

    private let client = APIClient()
    private var playbackStartTask: Task<PlaybackConfiguration, Error>?
    private var playbackSessionToRelease: (server: String, id: String)?
    private var playbackStartGeneration = 0
    private let logger = Logger(
        subsystem: Bundle.main.bundleIdentifier ?? "com.netv",
        category: "App"
    )

    init() {
        server = UserDefaults.standard.string(forKey: "server") ?? "http://localhost:8000"
        Task { await restoreSession() }
    }

    var filteredChannels: [ChannelRow] {
        guard !query.isEmpty else { return channels }
        return channels.filter {
            $0.channel.name.localizedCaseInsensitiveContains(query)
                || $0.programs.contains { $0.title.localizedCaseInsensitiveContains(query) }
        }
    }

    func filteredChannels(in categoryID: String?) -> [ChannelRow] {
        guard let categoryID else { return filteredChannels }
        let categoryIDs = guideCategoryGroups.first(where: { $0.id == categoryID })?.categoryIDs
            ?? [categoryID]
        return filteredChannels.filter { !categoryIDs.isDisjoint(with: $0.channel.categoryIDs) }
    }

    var guideCategoryGroups: [GuideCategoryGroup] {
        GuideCategoryGroup.distinct(guideCategories)
    }

    func signIn(username: String, password: String) async {
        logger.info("Sign-in started")
        isLoading = true
        errorMessage = nil
        defer { isLoading = false }
        do {
            _ = try await client.login(server: server, username: username, password: password)
            isAuthenticated = true
            logger.info("Sign-in succeeded")
            await loadGuide()
        } catch {
            logger.error("Sign-in failed: \(error.localizedDescription, privacy: .public)")
            errorMessage = connectionErrorMessage(for: error)
        }
    }

    func loadGuide(offset: Int? = nil) async {
        guard isAuthenticated else { return }
        let target = min(max(offset ?? requestedGuideOffset, -168), 168)
        requestedGuideOffset = target
        guideLoadGeneration += 1
        let generation = guideLoadGeneration
        isLoading = true
        errorMessage = nil
        defer {
            if generation == guideLoadGeneration {
                isLoading = false
                requestedGuideOffset = guideOffset
            }
        }
        do {
            let guide = try await client.guide(server: server, offset: target)
            guard generation == guideLoadGeneration, !Task.isCancelled else { return }
            channels = guide.rows
            guideCategories = guide.categories
            guideOffset = target
            guideWindowStart = Date(
                timeIntervalSince1970: guide.windowStartTimestamp
                    ?? floor(Date().timeIntervalSince1970 / 3600) * 3600 + Double(target) * 3600
            )
            if target == 0, selection == nil, let firstChannel = channels.first {
                play(firstChannel)
            }
            if channels.isEmpty {
                errorMessage = "No channels are selected. Choose guide categories in the neTV web settings."
            }
        } catch APIError.authenticationFailed {
            guard generation == guideLoadGeneration else { return }
            isAuthenticated = false
        } catch {
            guard generation == guideLoadGeneration, !Task.isCancelled else { return }
            errorMessage = error.localizedDescription
        }
    }

    func canPlayProgram(_ program: Program, in row: ChannelRow) -> Bool {
        !program.unavailable && (program.isCurrent
            || (program.catchup && row.channel.canCatchUp(from: program.startTimestamp)))
    }

    func isPlayingProgram(_ program: Program, in row: ChannelRow) -> Bool {
        guard let selection, selection.channel.id == row.id else { return false }
        if selection.isCatchup {
            guard let start = selection.program?.startTimestamp else { return false }
            return start == program.startTimestamp
        }
        return program.isCurrent
    }

    func playProgram(_ program: Program, in row: ChannelRow) {
        guard canPlayProgram(program, in: row) else {
            errorMessage = "This program is not available to play."
            return
        }
        if program.isCurrent {
            play(row)
        } else {
            playCatchup(row.channel, program: program)
        }
    }

    func play(_ row: ChannelRow) {
        selection = PlayerSelection(channel: row.channel, program: row.currentProgram)
    }

    func play(_ channel: Channel) {
        let row = channels.first { $0.id == channel.id }
        selection = PlayerSelection(channel: channel, program: row?.currentProgram)
    }

    /// Restarts what the selected live channel is airing, when its archive allows.
    var startOverForSelection: (() -> Void)? {
        guard let selection, !selection.isCatchup,
              let program = channels.first(where: { $0.id == selection.channel.id })?.currentProgram
                ?? selection.program,
              program.isCurrent, selection.channel.canCatchUp(from: program.startTimestamp) else { return nil }
        return { [weak self] in self?.playCatchup(selection.channel, program: program) }
    }

    /// Plays an archived program from its start.
    func playCatchup(_ channel: Channel, program: Program) {
        guard let start = program.startTimestamp else { return }
        selection = PlayerSelection(channel: channel, program: program, catchupStart: start)
    }

    func seekCatchup(
        _ playing: PlayerSelection, to timestamp: Double, resume: Bool,
        now: Double = Date().timeIntervalSince1970
    ) throws {
        guard selection?.id == playing.id else { return }
        guard playing.isCatchup, timestamp.isFinite, playing.channel.catchupDays > 0 else {
            throw APIError.server("This stream does not support archive seeking.")
        }
        if timestamp >= now - 30 {
            play(playing.channel)
            selection?.startPaused = !resume
            return
        }
        let oldest = now - Double(playing.channel.catchupDays) * 86_400 + 60
        let target = floor(max(timestamp, oldest))
        guard let end = playing.program?.endTimestamp, target < end else {
            throw APIError.server("This program is no longer available in the upstream archive.")
        }
        selection = PlayerSelection(
            channel: playing.channel, program: playing.program,
            catchupStart: target, startPaused: !resume
        )
    }

    func keepArchiveAlive(sessionID: String) async throws {
        try await client.keepArchiveAlive(server: server, sessionID: sessionID)
    }

    /// The program airing now, when the archive lets it restart from the beginning.
    func startOverProgram(for row: ChannelRow) -> Program? {
        guard let program = row.programs.first(where: \.isCurrent),
              row.channel.canCatchUp(from: program.startTimestamp) else { return nil }
        return program
    }

    func catchupPrograms(for channel: Channel) async throws -> [Program] {
        try await client.catchup(server: server, channelID: channel.id).programs
    }

    func playerConfiguration(for selection: PlayerSelection) async throws -> PlaybackConfiguration {
        let previous = playbackStartTask
        let requestServer = server
        playbackStartGeneration += 1
        let generation = playbackStartGeneration
        // Keep an in-flight start alive long enough to receive its session ID.
        // Cancelling its HTTP request doesn't cancel the server's encoder startup.
        let request = Task {
            if let previous { _ = try? await previous.value }
            if let previousSession = playbackSessionToRelease {
                try await client.releaseTranscode(server: previousSession.server, sessionID: previousSession.id)
                playbackSessionToRelease = nil
            }
            guard generation == playbackStartGeneration,
                  self.selection?.id == selection.id else {
                throw CancellationError()
            }
            let configuration = try await client.playbackConfiguration(
                server: requestServer, channelID: selection.channel.id,
                bandwidthSaver: bandwidthSaver, catchupStart: selection.catchupStart
            )
            if let sessionID = configuration.transcodeSessionID {
                playbackSessionToRelease = (requestServer, sessionID)
            }
            return configuration
        }
        playbackStartTask = request
        let configuration = try await request.value
        if Task.isCancelled {
            if let sessionID = configuration.transcodeSessionID {
                await client.stopTranscode(server: requestServer, sessionID: sessionID)
            }
            throw CancellationError()
        }
        return configuration
    }

    func reportPlaybackHealth(sessionID: String, health: PlaybackHealth) async throws -> PlaybackHealthResponse {
        let generation = playbackStartGeneration
        let response = try await client.reportPlaybackHealth(server: server, sessionID: sessionID, health: health)
        // Ignore late feedback from a replaced session. The active server can
        // both enable saver and clear it after sustained recovery.
        if playbackStartGeneration == generation, playbackSessionToRelease?.id == sessionID {
            bandwidthSaver = response.bandwidthSaver
        }
        return response
    }

    func stopPlayback(sessionID: String) async {
        await client.stopTranscode(server: server, sessionID: sessionID)
    }

    func signOut() async {
        await client.logout(server: server)
        channels = []
        guideCategories = []
        isAuthenticated = false
    }

    private func restoreSession() async {
        logger.info("Saved-session check started")
        defer { isCheckingSession = false }
        do {
            isAuthenticated = try await client.validateSession(server: server)
            logger.info("Saved-session check finished: authenticated=\(self.isAuthenticated)")
            isCheckingSession = false
            if isAuthenticated {
                await loadGuide()
            }
        } catch {
            isAuthenticated = false
            errorMessage = connectionErrorMessage(for: error)
            logger.error("Saved-session check failed: \(error.localizedDescription, privacy: .public)")
        }
    }

    private func connectionErrorMessage(for error: Error) -> String {
        guard let urlError = error as? URLError else {
            return error.localizedDescription
        }
        switch urlError.code {
        case .timedOut:
            return "The connection timed out. Check the server address, network or VPN connection, and Local Network access for neTV in Settings."
        case .cannotFindHost:
            return "The server name could not be resolved. Check the server address and DNS."
        case .cannotConnectToHost:
            return "The server refused the connection. Check that neTV is running and the port is correct."
        case .notConnectedToInternet:
            return "neTV cannot access the network. Check your Wi-Fi or VPN connection and allow Local Network access for neTV in Settings."
        case .secureConnectionFailed, .serverCertificateUntrusted,
             .serverCertificateHasBadDate, .serverCertificateHasUnknownRoot:
            return "A secure connection could not be established. Check the server certificate."
        default:
            return urlError.localizedDescription
        }
    }
}
