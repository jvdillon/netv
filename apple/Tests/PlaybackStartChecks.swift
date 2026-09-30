// Compile with Shared/Models.swift and Shared/AppModel.swift, excluding APIClient.swift.
// The fake transport holds a start open to reproduce channel changes during startup.
import Foundation
import Combine

struct PlaybackConfiguration {
    let url: URL
    let cookieHeader: String? = nil
    let transcodeSessionID: String?
}
struct PlaybackHealth {}
struct PlaybackHealthResponse { let bandwidthSaver: Bool }
enum APIError: Error { case authenticationFailed, server(String) }

@MainActor
final class APIClient {
    static var starts = 0
    static var active: Set<String> = []
    static var overlapping = false
    static var failRelease = false
    static var saverResponse = false
    static var requestedSaver: [Bool] = []
    static var requestedCatchup: [Double?] = []
    static var guideOffsets: [Int] = []
    static var guideResponses: [Int: String] = [:]
    static var guideDelays: [Int: UInt64] = [:]
    static var failingGuideOffset: Int?
    static var loginError: URLError?
    func playbackConfiguration(
        server: String, channelID: String, bandwidthSaver: Bool, catchupStart: Double?
    ) async throws -> PlaybackConfiguration {
        Self.requestedSaver.append(bandwidthSaver)
        Self.requestedCatchup.append(catchupStart)
        Self.starts += 1
        let id = "session-\(Self.starts)"
        Self.active.insert(id)
        Self.overlapping = Self.overlapping || Self.active.count > 1
        try await Task.sleep(for: .milliseconds(150))
        return PlaybackConfiguration(url: URL(string: "http://localhost/\(id)")!, transcodeSessionID: id)
    }
    func releaseTranscode(server: String, sessionID: String) async throws {
        if Self.failRelease { throw APIError.server("stop failed") }
        Self.active.remove(sessionID)
    }
    func stopTranscode(server: String, sessionID: String) async {
        try? await releaseTranscode(server: server, sessionID: sessionID)
    }
    func catchup(server: String, channelID: String) async throws -> CatchupListing {
        CatchupListing(days: 0, programs: [])
    }
    func keepArchiveAlive(server: String, sessionID: String) async throws {}
    func validateSession(server: String) async throws -> Bool { false }
    func login(server: String, username: String, password: String) async throws -> Bool {
        if let error = Self.loginError { throw error }
        return true
    }
    func logout(server: String) async {}
    func guide(server: String, offset: Int = 0) async throws -> GuideResponse {
        Self.guideOffsets.append(offset)
        if let delay = Self.guideDelays[offset] { try await Task.sleep(nanoseconds: delay) }
        if offset == Self.failingGuideOffset { throw APIError.server("Guide request failed") }
        return try JSONDecoder().decode(
            GuideResponse.self,
            from: Data((Self.guideResponses[offset] ?? "{\"rows\":[],\"categories\":[],\"total\":0}").utf8)
        )
    }
    func reportPlaybackHealth(server: String, sessionID: String, health: PlaybackHealth) async throws -> PlaybackHealthResponse {
        PlaybackHealthResponse(bandwidthSaver: Self.saverResponse)
    }
}

@main
struct PlaybackStartChecks {
    @MainActor
    static func main() async throws {
        let model = AppModel()
        try checkGuideFiltering(model)
        func selection(_ id: String) throws -> PlayerSelection {
            let data = Data("{\"stream_id\":\"\(id)\",\"name\":\"Test\",\"icon\":\"\"}".utf8)
            return PlayerSelection(channel: try JSONDecoder().decode(Channel.self, from: data), program: nil)
        }
        let a = try selection("a")
        let b = try selection("b")
        let c = try selection("c")
        model.selection = a
        let first = Task { try await model.playerConfiguration(for: a) }
        while APIClient.starts == 0 { await Task.yield() }
        first.cancel()
        model.selection = b
        let second = Task { try await model.playerConfiguration(for: b) }
        await Task.yield()
        second.cancel()
        model.selection = c
        let third = Task { try await model.playerConfiguration(for: c) }
        _ = try? await first.value
        _ = try? await second.value
        let current = try await third.value
        precondition(!APIClient.overlapping, "Channel changes opened overlapping provider sessions")
        precondition(APIClient.active == [current.transcodeSessionID!], "Cancelled startup leaked a session")

        APIClient.failRelease = true
        let startsBeforeFailure = APIClient.starts
        for _ in 0..<2 {
            do {
                _ = try await model.playerConfiguration(for: c)
                preconditionFailure("Started playback despite failing to release the old session")
            } catch {}
        }
        precondition(APIClient.starts == startsBeforeFailure)
        APIClient.failRelease = false
        let recovered = try await model.playerConfiguration(for: c)
        precondition(!APIClient.overlapping)
        precondition(APIClient.active == [recovered.transcodeSessionID!])
        let sessionID = recovered.transcodeSessionID!
        APIClient.saverResponse = true
        _ = try await model.reportPlaybackHealth(sessionID: sessionID, health: PlaybackHealth())
        precondition(model.bandwidthSaver, "Server fallback must enable saver")
        APIClient.saverResponse = false
        _ = try await model.reportPlaybackHealth(sessionID: "obsolete", health: PlaybackHealth())
        precondition(model.bandwidthSaver, "Obsolete session feedback must not clear saver")
        _ = try await model.reportPlaybackHealth(sessionID: sessionID, health: PlaybackHealth())
        precondition(!model.bandwidthSaver, "Server recovery must clear saver")
        let restored = try await model.playerConfiguration(for: c)
        precondition(APIClient.requestedSaver.last == false, "Recovery must restore normal startup")
        APIClient.saverResponse = true
        _ = try await model.reportPlaybackHealth(sessionID: sessionID, health: PlaybackHealth())
        precondition(!model.bandwidthSaver, "Late fallback must not affect the new session")
        await model.stopPlayback(sessionID: restored.transcodeSessionID!)
        precondition(APIClient.active.isEmpty)
        precondition(APIClient.requestedCatchup.allSatisfy { $0 == nil }, "Live tuning must not request the archive")

        let now = 1_700_010_000.0
        let archiveChannel = try JSONDecoder().decode(Channel.self, from: Data("""
            {"stream_id":"c","name":"Stream","icon":"","catchup_days":2}
            """.utf8))
        let archiveProgram = try JSONDecoder().decode(Program.self, from: Data("""
            {"title":"Recording","desc":"","start":"10:00","end":"11:00",
             "left_pct":0,"width_pct":100,"start_timestamp":1700000000,"end_timestamp":1700011000}
            """.utf8))
        let replay = PlayerSelection(channel: archiveChannel, program: archiveProgram, catchupStart: 1_700_000_000)
        model.selection = replay
        let archived = try await model.playerConfiguration(for: replay)
        precondition(APIClient.requestedCatchup.last == 1_700_000_000, "Catchup must request its program start")
        try model.seekCatchup(replay, to: 1_700_000_625.75, resume: false, now: now)
        let sought = model.selection!
        precondition(sought.catchupStart == 1_700_000_625 && sought.startPaused)
        precondition(sought.program == archiveProgram, "Reopening must retain the original program")
        let resumed = try await model.playerConfiguration(for: sought)
        precondition(APIClient.active == [resumed.transcodeSessionID!])
        precondition(!APIClient.overlapping, "Archive seeking must release the old upstream session")
        try model.seekCatchup(replay, to: 1_700_002_000, resume: true, now: now)
        precondition(model.selection == sought, "Ignore seek requests from a replaced player")
        try model.seekCatchup(sought, to: now - 5, resume: true, now: now)
        precondition(model.selection?.isCatchup == false, "Seeking to the live edge should return to live")
        model.selection = c
        let live = try await model.playerConfiguration(for: c)
        precondition(APIClient.requestedCatchup.last == .some(nil), "Going live must leave the archive")
        precondition(APIClient.active == [live.transcodeSessionID!], "Leaving catchup must release its session")
        precondition(archived.transcodeSessionID != live.transcodeSessionID)
        await model.stopPlayback(sessionID: live.transcodeSessionID!)
        try await checkGuideNavigation(model)
        for code in [URLError.Code.notConnectedToInternet, .timedOut] {
            APIClient.loginError = URLError(code)
            model.isAuthenticated = false
            await model.signIn(username: "test", password: "test")
            precondition(model.errorMessage?.contains("Local Network") == true)
            precondition(model.errorMessage?.contains("VPN") == true)
            precondition(!model.isLoading && !model.isAuthenticated)
        }
        APIClient.loginError = nil
        print("Playback startup checks passed: cancellation, rapid tuning, failed release, recovery, catchup")
    }

    @MainActor
    private static func checkGuideNavigation(_ model: AppModel) async throws {
        let now = Date().timeIntervalSince1970
        let hour = floor(now / 3600) * 3600
        func response(offset: Int, start: Double, end: Double, catchup: Bool, unavailable: Bool = false) throws -> String {
            let payload: [String: Any] = [
                "rows": [[
                    "channel": ["stream_id": "archive", "name": "Stream", "icon": "", "catchup_days": 2],
                    "programs": [[
                        "title": "Program", "desc": "", "start": "10:00", "end": "11:00",
                        "left_pct": 0, "width_pct": 50, "start_timestamp": start, "end_timestamp": end,
                        "catchup": catchup, "unavailable": unavailable,
                    ]],
                ]],
                "categories": [], "total": 1, "window_start_timestamp": hour + Double(offset) * 3600,
            ]
            return String(decoding: try JSONSerialization.data(withJSONObject: payload), as: UTF8.self)
        }
        APIClient.guideResponses = [
            0: try response(offset: 0, start: now - 900, end: now + 900, catchup: false),
            -3: try response(offset: -3, start: now - 7200, end: now - 3600, catchup: true),
            -6: try response(offset: -6, start: now - 21600, end: now - 18000, catchup: false, unavailable: true),
            3: try response(offset: 3, start: now + 10800, end: now + 14400, catchup: false),
        ]
        model.isAuthenticated = true
        await model.loadGuide(offset: 0)
        model.play(model.channels[0])
        let live = model.selection
        let starts = APIClient.starts
        await model.loadGuide(offset: -3)
        precondition(model.guideOffset == -3)
        precondition(model.guideWindowStart.timeIntervalSince1970 == hour - 10800)
        precondition(model.selection == live && APIClient.starts == starts, "Browsing must not retune playback")
        precondition(model.channels[0].currentProgram == nil, "An old listing is not the current live title")
        precondition(model.startOverForSelection != nil, "Browsing must preserve the live player's Start Over action")
        let row = model.channels[0]
        let past = row.programs[0]
        precondition(model.canPlayProgram(past, in: row))
        model.playProgram(past, in: row)
        precondition(model.selection?.catchupStart == past.startTimestamp, "Past program buttons must open the archive")
        precondition(model.isPlayingProgram(past, in: row))
        let archived = model.selection
        let undated = try JSONDecoder().decode(Program.self, from: Data("""
            {"title":"Undated","desc":"","start":"10:00","end":"11:00","left_pct":0,"width_pct":50}
            """.utf8))
        model.selection = PlayerSelection(channel: row.channel, program: nil, catchupStart: now - 7200)
        precondition(!model.isPlayingProgram(undated, in: row), "Missing timestamps cannot identify the playing program")
        model.selection = archived
        for offset in [-6, 3] {
            await model.loadGuide(offset: offset)
            let unavailableRow = model.channels[0]
            let program = unavailableRow.programs[0]
            precondition(!model.canPlayProgram(program, in: unavailableRow))
            model.playProgram(program, in: unavailableRow)
            precondition(model.selection == archived && model.errorMessage != nil, "Unavailable listings must never fall back to live")
        }
        await model.loadGuide(offset: 0)
        precondition(model.guideOffset == 0 && model.selection == archived, "Now changes the guide, not playback")
        APIClient.guideDelays[-3] = 100_000_000
        let count = APIClient.guideOffsets.count
        let olderRequest = Task { await model.loadGuide(offset: -3) }
        while APIClient.guideOffsets.count == count { await Task.yield() }
        await model.loadGuide(offset: 3)
        await olderRequest.value
        precondition(model.guideOffset == 3 && !model.isLoading, "A late response must not overwrite the latest window")
        await model.loadGuide(offset: -999)
        precondition(model.guideOffset == -168)
        await model.loadGuide(offset: 999)
        precondition(model.guideOffset == 168)
        APIClient.failingGuideOffset = 0
        await model.loadGuide(offset: 0)
        precondition(model.guideOffset == 168 && model.requestedGuideOffset == 168 && model.errorMessage != nil)
        APIClient.failingGuideOffset = nil
        APIClient.guideDelays = [:]
    }

    @MainActor
    private static func checkGuideFiltering(_ model: AppModel) throws {
        let data = Data("""
            [
              {"channel":{"stream_id":"news","name":"News","icon":"","category_ids":["1"]},
               "programs":[
                 {"title":"Headlines","desc":"","start":"00:00","end":"00:00","left_pct":0,"width_pct":50},
                 {"title":"Evening report","desc":"","start":"00:00","end":"00:00","left_pct":50,"width_pct":50}
               ]},
              {"channel":{"stream_id":"sports","name":"Sports","icon":"","category_ids":["2","1"]},"programs":[]},
              {"channel":{"stream_id":"other-news","name":"More News","icon":"","category_ids":["3"]},"programs":[]}
            ]
            """.utf8)
        model.channels = try JSONDecoder().decode([ChannelRow].self, from: data)
        model.guideCategories = try JSONDecoder().decode([GuideCategory].self, from: Data("""
            [
              {"category_id":"1","category_name":"News"},
              {"category_id":"2","category_name":"Sports"},
              {"category_id":"3","category_name":" news "}
            ]
            """.utf8))
        model.play(model.channels[0])
        let playing = model.selection
        defer {
            model.channels = []
            model.guideCategories = []
            model.query = ""
            model.selection = nil
        }
        precondition(model.filteredChannels(in: nil).count == 3)
        precondition(model.filteredChannels(in: "1").count == 2)
        precondition(model.filteredChannels(in: "2").map(\.id) == ["sports"])
        precondition(model.filteredChannels(in: "missing").isEmpty)
        precondition(model.guideCategoryGroups.count == 2)
        let newsGroup = model.guideCategoryGroups[0]
        precondition(model.filteredChannels(in: newsGroup.id).count == 3)
        model.query = "EVENING"
        precondition(model.filteredChannels(in: nil).map(\.id) == ["news"])
        precondition(model.filteredChannels(in: "2").isEmpty)
        precondition(model.filteredChannels(in: newsGroup.id).map(\.id) == ["news"])
        precondition(model.selection == playing, "Changing categories must not interrupt playback")
    }
}
