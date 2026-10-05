import Foundation

struct GuideResponse: Decodable {
    var rows: [ChannelRow]
    let categories: [GuideCategory]
    let total: Int
    let windowStartTimestamp: Double?

    enum CodingKeys: String, CodingKey {
        case rows, categories, total
        case windowStartTimestamp = "window_start_timestamp"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        rows = try container.decode([ChannelRow].self, forKey: .rows)
        categories = try container.decodeIfPresent([GuideCategory].self, forKey: .categories) ?? []
        total = try container.decode(Int.self, forKey: .total)
        windowStartTimestamp = try container.decodeIfPresent(Double.self, forKey: .windowStartTimestamp)
    }
}

struct GuideCategory: Decodable, Identifiable, Hashable {
    let id: String
    let name: String
    /// Shared playlists curated in the neTV web UI arrive as categories with kind "playlist".
    let isPlaylist: Bool

    enum CodingKeys: String, CodingKey {
        case id = "category_id"
        case name = "category_name"
        case kind
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(FlexibleID.self, forKey: .id).value
        name = try container.decode(String.self, forKey: .name)
        isPlaylist = try container.decodeIfPresent(String.self, forKey: .kind) == "playlist"
    }
}

struct GuideCategoryGroup: Identifiable, Hashable {
    let id: String
    let name: String
    var categoryIDs: Set<String>
    var isPlaylist = false

    static func distinct(_ categories: [GuideCategory]) -> [GuideCategoryGroup] {
        var groups: [GuideCategoryGroup] = []
        var indexByName: [String: Int] = [:]
        for category in categories {
            if category.isPlaylist {
                groups.append(GuideCategoryGroup(
                    id: "playlist:\(category.id)", name: category.name,
                    categoryIDs: [category.id], isPlaylist: true
                ))
                continue
            }
            let trimmed = category.name.trimmingCharacters(in: .whitespacesAndNewlines)
            let name = trimmed.isEmpty ? "Uncategorized" : trimmed
            let key = name.folding(options: .caseInsensitive, locale: Locale(identifier: "en_US_POSIX"))
            if let index = indexByName[key] {
                groups[index].categoryIDs.insert(category.id)
            } else {
                indexByName[key] = groups.count
                groups.append(GuideCategoryGroup(
                    id: "category:\(key)", name: name, categoryIDs: [category.id]
                ))
            }
        }
        return groups
    }
}

struct ChannelRow: Decodable, Identifiable, Hashable {
    let channel: Channel
    let programs: [Program]

    var id: String { channel.id }
    var currentProgram: Program? {
        programs.first(where: \.isCurrent)
    }
}

struct Channel: Decodable, Identifiable, Hashable {
    let streamID: FlexibleID
    let name: String
    let icon: String
    let categoryIDs: [String]
    /// Days of upstream archive; past programs within it can be watched from the start.
    let catchupDays: Int

    var id: String { streamID.value }

    enum CodingKeys: String, CodingKey {
        case streamID = "stream_id"
        case name
        case icon
        case categoryIDs = "category_ids"
        case catchupDays = "catchup_days"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        streamID = try container.decode(FlexibleID.self, forKey: .streamID)
        name = try container.decode(String.self, forKey: .name)
        icon = try container.decodeIfPresent(String.self, forKey: .icon) ?? ""
        let categories = try container.decodeIfPresent(
            [FlexibleID].self,
            forKey: .categoryIDs
        ) ?? []
        categoryIDs = categories.map(\.value)
        catchupDays = try container.decodeIfPresent(Int.self, forKey: .catchupDays) ?? 0
    }

    /// Whether a program starting at `start` is still in this channel's archive.
    func canCatchUp(from start: Double?, now: Date = Date()) -> Bool {
        guard catchupDays > 0, let start else { return false }
        let now = now.timeIntervalSince1970
        return start >= now - Double(catchupDays) * 86_400 && start < now
    }
}

struct Program: Decodable, Hashable {
    let title: String
    let desc: String
    let start: String
    let end: String
    let leftPercent: Double
    let widthPercent: Double
    let startTimestamp: Double?
    let endTimestamp: Double?
    /// The server confirmed this program can be watched from the upstream archive.
    let catchup: Bool
    let unavailable: Bool

    enum CodingKeys: String, CodingKey {
        case title, desc, start, end, catchup, unavailable
        case leftPercent = "left_pct"
        case widthPercent = "width_pct"
        case startTimestamp = "start_timestamp"
        case endTimestamp = "end_timestamp"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        title = try container.decode(String.self, forKey: .title)
        desc = try container.decode(String.self, forKey: .desc)
        start = try container.decode(String.self, forKey: .start)
        end = try container.decode(String.self, forKey: .end)
        leftPercent = try container.decode(Double.self, forKey: .leftPercent)
        widthPercent = try container.decode(Double.self, forKey: .widthPercent)
        startTimestamp = try container.decodeIfPresent(Double.self, forKey: .startTimestamp)
        endTimestamp = try container.decodeIfPresent(Double.self, forKey: .endTimestamp)
        catchup = try container.decodeIfPresent(Bool.self, forKey: .catchup) ?? false
        unavailable = try container.decodeIfPresent(Bool.self, forKey: .unavailable) ?? false
    }

    /// Day and time, for listings that reach back past today.
    var archiveLabel: String {
        guard let startTimestamp else { return timeRange }
        let startDate = Date(timeIntervalSince1970: startTimestamp)
        let day = Calendar.current.isDateInToday(startDate)
            ? "Today"
            : startDate.formatted(.dateTime.weekday(.abbreviated).month(.abbreviated).day())
        return "\(day), \(timeRange)"
    }

    var timeRange: String {
        guard let startTimestamp, let endTimestamp else { return "\(start) – \(end)" }
        let startDate = Date(timeIntervalSince1970: startTimestamp)
        let endDate = Date(timeIntervalSince1970: endTimestamp)
        return "\(startDate.formatted(date: .omitted, time: .shortened)) – \(endDate.formatted(date: .omitted, time: .shortened))"
    }

    var guideRange: ClosedRange<Double> {
        let lower = min(max(leftPercent / 100, 0), 1)
        let upper = min(max((leftPercent + widthPercent) / 100, lower), 1)
        return lower...upper
    }

    var isCurrent: Bool {
        if let startTimestamp, let endTimestamp {
            let now = Date().timeIntervalSince1970
            return now >= startTimestamp && now < endTimestamp
        }
        guard let startDate = Self.timeFormatter.date(from: start),
              let endDate = Self.timeFormatter.date(from: end) else { return false }
        let now = Date()
        let calendar = Calendar.current
        let startComponents = calendar.dateComponents([.hour, .minute], from: startDate)
        let endComponents = calendar.dateComponents([.hour, .minute], from: endDate)
        guard let todayStart = calendar.date(
            bySettingHour: startComponents.hour ?? 0,
            minute: startComponents.minute ?? 0,
            second: 0,
            of: now
        ), var todayEnd = calendar.date(
            bySettingHour: endComponents.hour ?? 0,
            minute: endComponents.minute ?? 0,
            second: 0,
            of: now
        ) else { return false }
        if todayEnd <= todayStart {
            todayEnd = calendar.date(byAdding: .day, value: 1, to: todayEnd) ?? todayEnd
        }
        return now >= todayStart && now < todayEnd
    }

    var progress: Double {
        if let startTimestamp, let endTimestamp {
            guard isCurrent, endTimestamp > startTimestamp else { return 0 }
            return min(max((Date().timeIntervalSince1970 - startTimestamp) / (endTimestamp - startTimestamp), 0), 1)
        }
        guard isCurrent,
              let startDate = Self.timeFormatter.date(from: start),
              let endDate = Self.timeFormatter.date(from: end) else { return 0 }
        let calendar = Calendar.current
        let now = Date()
        let startComponents = calendar.dateComponents([.hour, .minute], from: startDate)
        let endComponents = calendar.dateComponents([.hour, .minute], from: endDate)
        guard let todayStart = calendar.date(
            bySettingHour: startComponents.hour ?? 0,
            minute: startComponents.minute ?? 0,
            second: 0,
            of: now
        ), var todayEnd = calendar.date(
            bySettingHour: endComponents.hour ?? 0,
            minute: endComponents.minute ?? 0,
            second: 0,
            of: now
        ) else { return 0 }
        if todayEnd <= todayStart {
            todayEnd = calendar.date(byAdding: .day, value: 1, to: todayEnd) ?? todayEnd
        }
        return min(max(now.timeIntervalSince(todayStart) / todayEnd.timeIntervalSince(todayStart), 0), 1)
    }

    private static let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "HH:mm"
        return formatter
    }()
}

struct UserPreferences: Decodable {
    let guideFilter: [String]?

    enum CodingKeys: String, CodingKey {
        case guideFilter = "guide_filter"
    }
}

struct FlexibleID: Decodable, Hashable {
    let value: String

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let string = try? container.decode(String.self) {
            value = string
        } else {
            value = String(try container.decode(Int.self))
        }
    }
}

struct CatchupListing: Decodable {
    let days: Int
    let programs: [Program]
}

struct PlaybackCaptionChoice: Identifiable, Equatable {
    let id: String
    let title: String
}

struct PlaybackCaptionSelectionRequest: Equatable {
    let token = UUID()
    let choiceID: String?
}

struct PlayerSelection: Identifiable, Hashable {
    let channel: Channel
    let program: Program?
    /// Requested Unix position within the upstream archive; nil plays live.
    var catchupStart: Double?
    var startPaused = false

    var isCatchup: Bool { catchupStart != nil }

    /// Distinct per archive position so seeking outside the current media restarts playback.
    var id: String {
        guard let catchupStart else { return channel.id }
        return "\(channel.id)@\(Int(catchupStart))"
    }
}

struct ArchiveTimeline {
    let start: Double
    let end: Double
    let streamStart: Double

    init?(selection: PlayerSelection, streamStart: Double? = nil) {
        guard let requested = selection.catchupStart, requested.isFinite,
              let start = selection.program?.startTimestamp, start.isFinite,
              let end = selection.program?.endTimestamp, end.isFinite, end > start else { return nil }
        self.start = start
        self.end = end
        self.streamStart = streamStart ?? floor(requested / 60) * 60
    }

    var duration: Double { end - start }

    func elapsed(mediaTime: Double) -> Double {
        guard mediaTime.isFinite else { return 0 }
        return min(max(streamStart + mediaTime - start, 0), duration)
    }

    func timestamp(elapsed: Double) -> Double {
        start + min(max(elapsed, 0), max(0, duration - 1))
    }

    func localPosition(elapsed: Double, seekable: [ClosedRange<Double>]) -> Double? {
        let position = timestamp(elapsed: elapsed) - streamStart
        guard position >= 0, seekable.contains(where: {
            position >= $0.lowerBound && position < $0.upperBound
        }) else { return nil }
        return position
    }
}

struct LiveTimeline {
    let start: Double
    let end: Double
    let position: Double

    init?(
        seekable: [ClosedRange<Double>], position: Double, maximumDuration: Double
    ) {
        guard position.isFinite, maximumDuration.isFinite, maximumDuration > 0,
              let latest = seekable.last(where: {
                $0.lowerBound.isFinite && $0.upperBound.isFinite
                    && $0.upperBound > $0.lowerBound
              }) else { return nil }
        let duration = min(maximumDuration, 2 * 60 * 60)
        start = max(latest.lowerBound, latest.upperBound - duration)
        end = latest.upperBound
        guard end > start else { return nil }
        self.position = min(max(position, start), end)
    }

    var duration: Double { end - start }
    var elapsed: Double { position - start }
    var behindLive: Double { end - position }
    var isAtLiveEdge: Bool { behindLive <= 15 }

    func seekTarget(offsetBy seconds: Double) -> Double {
        let margin = min(0.1, duration / 2)
        return min(max(position + seconds, start + margin), end - margin)
    }
}

struct LiveProgramTimeline {
    let start: Double
    let end: Double
    let playback: Double
    let live: Double

    init?(
        program: Program?, playbackTimestamp: Double, liveTimestamp: Double
    ) {
        guard let start = program?.startTimestamp, start.isFinite,
              let end = program?.endTimestamp, end.isFinite, end > start,
              playbackTimestamp.isFinite, liveTimestamp.isFinite,
              liveTimestamp >= start, liveTimestamp <= end else { return nil }
        self.start = start
        self.end = end
        self.live = min(max(liveTimestamp, start), end)
        self.playback = min(max(playbackTimestamp, start), self.live)
    }

    var duration: Double { end - start }
    var playbackProgress: Double { (playback - start) / duration }
    var liveProgress: Double { (live - start) / duration }
    var behindLive: Double { live - playback }
}
