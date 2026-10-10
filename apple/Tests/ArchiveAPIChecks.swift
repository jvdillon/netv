// Compile with Shared/Models.swift and Shared/APIClient.swift.
import Foundation

private final class ArchiveTransport: URLProtocol {
    static var requests: [URLRequest] = []
    static var redirectToLogin = false
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.requests.append(request)
        let url = request.url!
        let body: String
        if url.path.hasPrefix("/play/live/") {
            body = """
                rawUrl: "https://upstream.test/timeshift/user/pass/60/2026-09-28:10-00/1.ts",
                transcodeMode: "always", sourceId: "test", deinterlaceFallback: false,
                catchupStart: 1700000580.0, catchupSeek: 45.0, liveDvrMins: 999
                """
        } else if url.path == "/api/user-prefs" {
            body = #"{"guide_filter":[1,{"invalid":true}]}"#
        } else if url.path == "/api/guide/rows" {
            let query = URLComponents(url: url, resolvingAgainstBaseURL: false)!.queryItems!
            let start = query.first { $0.name == "start" }!.value!
            let offset = Int(query.first { $0.name == "offset" }!.value!)!
            body = """
                {"rows":[{"channel":{"stream_id":"\(start)","name":"Stream","icon":""},"programs":[]}],
                 "total":2,"window_start_timestamp":\(1700000000 + offset * 3600)}
                """
        } else if url.path == "/transcode/start" {
            let contentType = URLComponents(
                url: url, resolvingAgainstBaseURL: false
            )?.queryItems?.first(where: { $0.name == "content_type" })?.value
            if contentType == "movie" {
                body = """
                    {"session_id":"archive","playlist":"/transcode/archive/stream.m3u8",
                     "caption_playlist":"/transcode/archive/captions.m3u8"}
                    """
            } else {
                body = """
                    {"session_id":"live","playlist":"/transcode/live/low.m3u8",
                     "caption_playlist":"/transcode/live/low-captions.m3u8",
                     "playback_buffer_seconds":30}
                    """
            }
        } else if url.path == "/transcode/live/health" {
            body = """
                {"bandwidth_saver":false,"playlist":"/transcode/live/high.m3u8",
                 "caption_playlist":"/transcode/live/high-captions.m3u8"}
                """
        } else if url.path.hasPrefix("/transcode/progress/") {
            body = #"{"duration":120,"segment_count":60}"#
        } else {
            body = #"{"status":"stopped"}"#
        }
        let responseURL = Self.redirectToLogin
            ? URL(string: "http://netv.test/login")!
            : url
        client?.urlProtocol(self, didReceive: HTTPURLResponse(
            url: responseURL, statusCode: 200, httpVersion: nil, headerFields: nil
        )!, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(body.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

@main
struct ArchiveAPIChecks {
    static func main() async throws {
        let defaults = APIClient.sessionConfiguration()
        precondition(!defaults.waitsForConnectivity, "Unavailable network access must not hold the startup screen")
        precondition(defaults.timeoutIntervalForResource == 60, "Requests must have a bounded overall timeout")
        precondition(defaults.timeoutIntervalForRequest == 60 && defaults.httpShouldSetCookies)
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [ArchiveTransport.self]
        let client = APIClient(session: URLSession(configuration: configuration))
        ArchiveTransport.redirectToLogin = true
        let savedSessionIsValid = try await client.validateSession(server: "http://netv.test")
        precondition(
            !savedSessionIsValid,
            "A login redirect must be treated as an expired saved session"
        )
        do {
            _ = try await client.guide(server: "http://netv.test")
            preconditionFailure("A login redirect must not be decoded as guide JSON")
        } catch APIError.authenticationFailed {
        }
        ArchiveTransport.redirectToLogin = false
        ArchiveTransport.requests = []
        let playback = try await client.playbackConfiguration(
            server: "http://netv.test", channelID: "1", bandwidthSaver: true,
            catchupStart: 1_700_000_625
        )
        precondition(playback.archiveStart == 1_700_000_580)
        precondition(playback.archiveSeek == 45)
        precondition(playback.transcodeSessionID == "archive")
        precondition(playback.url.path == "/transcode/archive/captions.m3u8")
        precondition(playback.playbackBufferDuration == 12)
        let start = ArchiveTransport.requests.first { $0.url?.path == "/transcode/start" }!
        let query = URLComponents(url: start.url!, resolvingAgainstBaseURL: false)!.queryItems!
        precondition(query.contains(URLQueryItem(name: "content_type", value: "movie")))
        precondition(query.contains(URLQueryItem(name: "fast_start", value: "false")))
        precondition(query.contains(URLQueryItem(name: "bandwidth_saver", value: "false")))
        try await client.keepArchiveAlive(server: "http://netv.test", sessionID: "archive")
        precondition(ArchiveTransport.requests.last?.url?.path == "/transcode/progress/archive")
        try await client.releaseTranscode(server: "http://netv.test", sessionID: "archive")
        precondition(ArchiveTransport.requests.last?.httpMethod == "DELETE")
        precondition(ArchiveTransport.requests.last?.url?.query == "force=true")
        ArchiveTransport.requests = []
        let live = try await client.playbackConfiguration(
            server: "http://netv.test", channelID: "1"
        )
        precondition(live.liveBufferDuration == 7200)
        precondition(live.playbackBufferDuration == 30)
        precondition(live.url.path == "/transcode/live/low-captions.m3u8")
        let liveStart = ArchiveTransport.requests.first { $0.url?.path == "/transcode/start" }!
        let liveQuery = URLComponents(
            url: liveStart.url!, resolvingAgainstBaseURL: false
        )!.queryItems!
        precondition(liveQuery.contains(URLQueryItem(name: "content_type", value: "live")))
        precondition(liveQuery.contains(URLQueryItem(name: "fast_start", value: "true")))
        let health = try await client.reportPlaybackHealth(
            server: "http://netv.test",
            sessionID: "live",
            health: PlaybackHealth(
                bufferSeconds: 10, waiting: false,
                observedBitrate: 10_000_000, requiredBitrate: 4_000_000
            )
        )
        precondition(health.playlist == "/transcode/live/high.m3u8")
        precondition(health.captionPlaylist == "/transcode/live/high-captions.m3u8")
        for offset in [-6, 0, 3] {
            ArchiveTransport.requests = []
            let guide = try await client.guide(server: "http://netv.test", offset: offset)
            precondition(guide.rows.count == 2)
            let pages = ArchiveTransport.requests.filter { $0.url?.path == "/api/guide/rows" }
            precondition(pages.count == 2)
            precondition(
                !ArchiveTransport.requests.contains { $0.url?.path == "/api/user-prefs" },
                "Guide loading must rely on the server-side saved filter"
            )
            for page in pages {
                let query = URLComponents(url: page.url!, resolvingAgainstBaseURL: false)!.queryItems!
                precondition(query.contains(URLQueryItem(name: "offset", value: String(offset))))
                precondition(query.contains(URLQueryItem(name: "cats", value: "")))
            }
        }
        print("Archive API checks passed: exact position, VOD mode, heartbeat and forced release")
    }
}
