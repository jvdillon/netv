import Foundation

@main
struct GuideDataChecks {
    static func main() throws {
        let decoder = JSONDecoder()
        let now = Date().timeIntervalSince1970
        func program(left: Double, width: Double, start: Double? = nil, end: Double? = nil) throws -> Program {
            var payload: [String: Any] = [
                "title": "News", "desc": "", "start": "00:00", "end": "00:00",
                "left_pct": left, "width_pct": width
            ]
            if let start { payload["start_timestamp"] = start }
            if let end { payload["end_timestamp"] = end }
            return try decoder.decode(
                Program.self, from: JSONSerialization.data(withJSONObject: payload)
            )
        }

        let current = try program(left: 0, width: 50, start: now - 1800, end: now + 1800)
        precondition(current.isCurrent, "Absolute broadcast times must not depend on the device time zone")
        precondition(abs(current.progress - 0.5) < 0.01)
        let upcoming = try program(left: 50, width: 50, start: now + 1800, end: now + 3600)
        precondition(!upcoming.isCurrent && upcoming.progress == 0)
        let ended = try program(left: 0, width: 10, start: now - 3600, end: now - 1800)
        precondition(!ended.isCurrent)

        let clippedLeft = try program(left: -10, width: 30)
        precondition(clippedLeft.guideRange == (0...0.2))
        let clippedRight = try program(left: 90, width: 40)
        precondition(clippedRight.guideRange == (0.9...1))
        let shortListing = try program(left: 10, width: 1)
        precondition(abs(shortListing.guideRange.upperBound - shortListing.guideRange.lowerBound - 0.01) < 0.0001)
        let offscreen = try program(left: 110, width: 10)
        precondition(offscreen.guideRange == (1...1))

        let legacy = try decoder.decode(GuideResponse.self, from: Data("""
            {"rows":[{"channel":{"stream_id":1,"name":"News","icon":""},"programs":[]}],"total":1}
            """.utf8))
        precondition(legacy.categories.isEmpty && legacy.windowStartTimestamp == nil)
        precondition(legacy.rows[0].channel.categoryIDs.isEmpty)
        precondition(clippedLeft.timeRange == "00:00 \u{2013} 00:00")

        let categorized = try decoder.decode(GuideResponse.self, from: Data("""
            {
              "rows":[{"channel":{"stream_id":"1","name":"News","icon":"","category_ids":[1,"2"]},"programs":[]}],
              "categories":[{"category_id":1,"category_name":"News"},{"category_id":"2","category_name":"Sports"}],
              "total":1,"window_start_timestamp":1789344000
            }
            """.utf8))
        precondition(categorized.categories.map(\.id) == ["1", "2"])
        precondition(categorized.rows[0].channel.categoryIDs == ["1", "2"])
        precondition(categorized.windowStartTimestamp == 1789344000)
        let duplicates = try decoder.decode([GuideCategory].self, from: Data("""
            [
              {"category_id":"sport-1","category_name":"Sports"},
              {"category_id":"news-1","category_name":"News"},
              {"category_id":"sport-2","category_name":" sports "},
              {"category_id":"sport-1","category_name":"Sports"},
              {"category_id":"news-2","category_name":"NEWS"}
            ]
            """.utf8))
        let groups = GuideCategoryGroup.distinct(duplicates)
        precondition(groups.map(\.name) == ["Sports", "News"], "Preserve category order without duplicate rows")
        precondition(groups[0].categoryIDs == ["sport-1", "sport-2"])
        precondition(groups[1].categoryIDs == ["news-1", "news-2"])
        precondition(Set(groups.map(\.id)).count == groups.count)
        precondition(GuideCategoryGroup.distinct([]).isEmpty)

        precondition(!legacy.rows[0].channel.canCatchUp(from: now - 60), "Older servers report no archive")
        precondition(!current.catchup && !current.unavailable)
        let archived = try decoder.decode(GuideResponse.self, from: Data("""
            {
              "rows":[{"channel":{"stream_id":"7","name":"News","catchup_days":2},
                       "programs":[{"title":"Earlier","desc":"","start":"09:00","end":"10:00",
                                    "left_pct":0,"width_pct":20,"start_timestamp":\(now - 7200),
                                    "end_timestamp":\(now - 3600),"catchup":true}]}],
              "total":1
            }
            """.utf8))
        let archive = archived.rows[0].channel
        precondition(archive.catchupDays == 2 && archived.rows[0].programs[0].catchup)
        precondition(archived.rows[0].currentProgram == nil, "Past guide windows must not supply live metadata")
        let unavailable = try decoder.decode(Program.self, from: Data("""
            {"title":"Unavailable","desc":"","start":"09:00","end":"10:00",
             "left_pct":0,"width_pct":20,"unavailable":true}
            """.utf8))
        precondition(unavailable.unavailable)
        precondition(archive.canCatchUp(from: now - 86_400))
        precondition(!archive.canCatchUp(from: now - 3 * 86_400), "Programs older than the archive can't restart")
        precondition(!archive.canCatchUp(from: now + 60) && !archive.canCatchUp(from: nil))

        let live = PlayerSelection(channel: archive, program: nil)
        let replay = PlayerSelection(channel: archive, program: nil, catchupStart: now - 7200)
        precondition(!live.isCatchup && replay.isCatchup)
        precondition(live.id == "7" && replay.id != live.id, "Archived playback must restart the player")
        let base = floor(now / 60) * 60 - 7200
        let recording = try program(left: 0, width: 100, start: base, end: base + 3600)
        let sought = PlayerSelection(channel: archive, program: recording, catchupStart: base + 625)
        let timeline = ArchiveTimeline(selection: sought)!
        precondition(timeline.streamStart == base + 600, "Archives start on whole minutes")
        precondition(timeline.elapsed(mediaTime: 25) == 625, "The timeline must include the skipped portion")
        precondition(timeline.duration == 3600, "Reopening must retain the full program duration")
        precondition(timeline.localPosition(elapsed: 650, seekable: [0...120]) == 50)
        precondition(timeline.localPosition(elapsed: 300, seekable: [0...120]) == nil, "Earlier media must reopen")
        precondition(timeline.localPosition(elapsed: 1200, seekable: [0...120]) == nil, "Unproduced media must reopen")
        precondition(timeline.localPosition(elapsed: 660, seekable: [0...30, 90...120]) == nil, "Do not seek into a media gap")
        precondition(timeline.timestamp(elapsed: -10) == base)
        precondition(timeline.timestamp(elapsed: 4000) == base + 3599, "Do not accidentally open the next program")
        precondition(timeline.elapsed(mediaTime: .nan) == 0)
        precondition(ArchiveTimeline(selection: live) == nil)
        let adjusted = ArchiveTimeline(selection: sought, streamStart: base + 590)!
        precondition(adjusted.elapsed(mediaTime: 35) == 625, "Use the server's actual archive start")

        let liveTimeline = LiveTimeline(
            seekable: [100...460], position: 430, maximumDuration: 7200
        )!
        precondition(liveTimeline.duration == 360 && liveTimeline.elapsed == 330)
        precondition(liveTimeline.behindLive == 30 && !liveTimeline.isAtLiveEdge)
        precondition(liveTimeline.seekTarget(offsetBy: -15) == 415)
        precondition(liveTimeline.seekTarget(offsetBy: 15) == 445)
        precondition(
            LiveTimeline(seekable: [100...460], position: 450, maximumDuration: 7200)!
                .isAtLiveEdge
        )
        let cappedTimeline = LiveTimeline(
            seekable: [0...10_000], position: 100, maximumDuration: 99_999
        )!
        precondition(cappedTimeline.start == 2800 && cappedTimeline.duration == 7200)
        precondition(
            cappedTimeline.seekTarget(offsetBy: -15) == 2800.1,
            "Seeking must stay inside the retained live window"
        )
        precondition(
            LiveTimeline(seekable: [], position: 0, maximumDuration: 7200) == nil
        )
        print("Guide checks passed: guide data, archive and live timelines")
    }
}
