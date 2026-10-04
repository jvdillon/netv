import SwiftUI
#if os(macOS)
import AppKit
#endif

struct WatchView: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.scenePhase) private var scenePhase
    #if os(macOS) || os(tvOS)
    @State private var selectedCategoryID: String?
    #endif
    #if os(macOS)
    @FocusState private var isExpandButtonFocused: Bool
    #endif
    #if os(tvOS)
    @FocusState private var isPlayerFocused: Bool
    #endif

    var body: some View {
        Group {
            #if os(macOS) || os(tvOS)
            largeScreenLayout
            #elseif os(iOS)
            standardLayout
                .fullScreenCover(isPresented: $model.isPlayerExpanded) {
                    iOSFullScreenPlayer
                }
            #else
            standardLayout
            #endif
        }
        .task { await refreshGuideIfVisible() }
        .onChange(of: scenePhase) { _, phase in
            guard phase == .active else { return }
            Task { await refreshGuideIfVisible() }
        }
        .onChange(of: model.isPlayerExpanded) { _, expanded in
            #if os(tvOS)
            isPlayerFocused = expanded
            #endif
            guard !expanded else { return }
            Task { await refreshGuideIfVisible() }
        }
        #if os(macOS)
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            Task { await refreshGuideIfVisible() }
        }
        #endif
    }

    #if os(macOS) || os(tvOS)
    private var largeScreenLayout: some View {
        GeometryReader { proxy in
            let sidebarWidth = min(
                GuideMetrics.scaled(250), max(GuideMetrics.scaled(200), proxy.size.width * 0.21)
            )
            let contentWidth = max(proxy.size.width - sidebarWidth, 0)
            let heroHeight = min(
                GuideMetrics.scaled(240), max(GuideMetrics.scaled(180), proxy.size.height * 0.29)
            )
            let previewWidth = min(contentWidth * 0.42, heroHeight * 16 / 9)
            ZStack(alignment: .topLeading) {
                HStack(spacing: 0) {
                    GuideSidebar(selectedCategoryID: $selectedCategoryID)
                        .frame(width: sidebarWidth)
                    VStack(spacing: 0) {
                        HStack(spacing: 0) {
                            GuideHeroInfo(selection: currentSelection)
                                .frame(width: contentWidth - previewWidth, height: heroHeight)
                            Color.clear
                                .frame(width: previewWidth, height: heroHeight)
                        }
                        GuideTheme.divider.frame(height: 1)
                        ChannelGuideView(selectedCategoryID: selectedCategoryID)
                            .frame(
                                width: contentWidth,
                                height: max(proxy.size.height - heroHeight - 1, 0)
                            )
                    }
                }
                .opacity(model.isPlayerExpanded ? 0 : 1)
                .allowsHitTesting(!model.isPlayerExpanded)
                .disabled(model.isPlayerExpanded)
                .accessibilityHidden(model.isPlayerExpanded)

                // Keep one player mounted while its frame changes, avoiding a stream retune.
                ZStack(alignment: .topTrailing) {
                    playerSurface
                    #if os(tvOS)
                    // Focus changes can rebuild their subtree. Keep the playback
                    // task outside that subtree when switching presentation modes.
                    Color.clear
                        .contentShape(Rectangle())
                        .focusable(model.isPlayerExpanded)
                        .focused($isPlayerFocused)
                        .onTapGesture { model.playPauseRequest = UUID() }
                        .onMoveCommand { direction in
                            model.playerActivity = UUID()
                            if direction == .left { model.seekBackwardRequest = UUID() }
                            if direction == .right { model.seekForwardRequest = UUID() }
                        }
                    #endif
                    #if os(macOS)
                    expandButton
                    #endif
                }
                .frame(
                    width: model.isPlayerExpanded ? proxy.size.width : previewWidth,
                    height: model.isPlayerExpanded ? proxy.size.height : heroHeight
                )
                .background(.black)
                .clipped()
                .offset(x: model.isPlayerExpanded ? 0 : proxy.size.width - previewWidth)
            }
            .frame(width: proxy.size.width, height: proxy.size.height, alignment: .topLeading)
            .background(GuideTheme.background)
        }
        #if os(macOS)
        .toolbar(model.isPlayerExpanded ? .hidden : .automatic)
        .onReceive(NotificationCenter.default.publisher(for: NSWindow.didExitFullScreenNotification)) { _ in
            withAnimation(.easeInOut(duration: 0.2)) {
                model.isPlayerExpanded = false
            }
        }
        #endif
    }

    private var currentSelection: PlayerSelection? {
        guard let selection = model.selection else { return nil }
        // Archived playback keeps the program it was started for.
        guard !selection.isCatchup,
              let row = model.channels.first(where: { $0.id == selection.channel.id }),
              let program = row.currentProgram else {
            return selection
        }
        return PlayerSelection(channel: row.channel, program: program)
    }
    #endif

    private func refreshGuideIfVisible() async {
        guard !model.isPlayerExpanded else { return }
        await model.refreshGuideOnReturn()
    }

    private var standardLayout: some View {
        GeometryReader { proxy in
            VStack(spacing: 0) {
                ZStack(alignment: .topTrailing) {
                    #if os(iOS)
                    if model.isPlayerExpanded {
                        Color.black
                    } else {
                        playerSurface
                        expandButton
                    }
                    #else
                    playerSurface
                    expandButton
                    #endif
                }
                .frame(
                    width: proxy.size.width,
                    height: model.isPlayerExpanded ? proxy.size.height : playerHeight(for: proxy.size)
                )
                .background(.black)
                .clipped()

                if !model.isPlayerExpanded {
                    Rectangle()
                        .fill(Color.white.opacity(0.08))
                        .frame(height: 1)
                    GuideView()
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .background(Theme.backgroundGradient)
        }
        .navigationTitle(model.isPlayerExpanded ? "" : "Live TV")
    }

    #if os(iOS)
    private var iOSFullScreenPlayer: some View {
        ZStack(alignment: .topTrailing) {
            Color.black
            if let selection = model.selection {
                PlayerView(selection: selection)
                    .id(selection.id)
            }
            Button {
                model.isPlayerExpanded = false
            } label: {
                Image(systemName: "arrow.down.right.and.arrow.up.left")
                    .font(.headline)
                    .padding(11)
                    .background(.black.opacity(0.62), in: Circle())
            }
            .buttonStyle(.plain)
            .foregroundStyle(.white)
            .padding(16)
            .accessibilityLabel("Exit full screen")
        }
        .background(Color.black)
        .ignoresSafeArea()
        .statusBarHidden(true)
        .persistentSystemOverlays(.hidden)
        .interactiveDismissDisabled()
    }
    #endif

    @ViewBuilder
    private var expandButton: some View {
        if model.selection != nil {
            Button {
                togglePlayerSize()
            } label: {
                Label(
                    model.isPlayerExpanded ? "Show Guide" : "Full Screen",
                    systemImage: model.isPlayerExpanded
                        ? "arrow.down.right.and.arrow.up.left"
                        : "arrow.up.left.and.arrow.down.right"
                )
                .labelStyle(.iconOnly)
                .font(.headline)
                .padding(11)
                .background(.black.opacity(0.62), in: Circle())
            }
            #if os(macOS)
            .buttonStyle(GuideButtonStyle(isFocused: isExpandButtonFocused))
            .focused($isExpandButtonFocused)
            #else
            .buttonStyle(.plain)
            #endif
            .foregroundStyle(.white)
            #if os(tvOS)
            .padding(model.isPlayerExpanded ? 48 : 16)
            #else
            .padding(16)
            #endif
            .accessibilityLabel(model.isPlayerExpanded ? "Exit full screen" : "Full screen")
        }
    }

    @ViewBuilder
    private var playerSurface: some View {
        if let selection = model.selection {
            PlayerView(selection: selection)
                .id(selection.id)
        } else {
            VStack(spacing: 14) {
                Image(systemName: "play.tv.fill")
                    .font(.system(size: 48))
                    .foregroundStyle(Theme.accent)
                Text("Choose a channel")
                    .font(.title2.bold())
                Text("Your live program will play here.")
                    .foregroundStyle(Theme.secondaryText)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(
                LinearGradient(
                    colors: [Theme.surface, .black],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            )
        }
    }

    #if os(macOS) || os(tvOS)
    private struct GuideHeroInfo: View {
        let selection: PlayerSelection?

        var body: some View {
            VStack(alignment: .leading, spacing: 14) {
                if let selection {
                    HStack(alignment: .center, spacing: 16) {
                        ChannelLogo(channel: selection.channel, size: GuideMetrics.scaled(64))
                        VStack(alignment: .leading, spacing: 7) {
                            Text(selection.channel.name)
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(Theme.secondaryText)
                                .lineLimit(1)
                            Text(selection.program?.title ?? "Live TV")
                                .font(.system(size: GuideMetrics.fontSize(24), weight: .bold))
                                .lineLimit(2)
                            HStack(spacing: 10) {
                                if selection.isCatchup { CatchupBadge() } else { LiveBadge() }
                                if let program = selection.program {
                                    Text(program.timeRange)
                                        .font(.caption.weight(.medium))
                                        .lineLimit(1)
                                }
                            }
                        }
                    }
                    if let program = selection.program, !program.desc.isEmpty {
                        Text(program.desc)
                            .font(.callout)
                            .foregroundStyle(Theme.secondaryText)
                            .lineLimit(2)
                    }
                } else {
                    Text("Live TV")
                        .font(.system(size: GuideMetrics.fontSize(26), weight: .bold))
                    Text("Choose a channel from the guide to start watching.")
                        .foregroundStyle(Theme.secondaryText)
                }
            }
            .padding(GuideMetrics.scaled(24))
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
            .background(GuideTheme.background)
        }
    }
    #endif

    private func playerHeight(for size: CGSize) -> CGFloat {
        #if os(tvOS)
        min(size.height * 0.36, 360)
        #elseif os(macOS)
        min(size.height * 0.58, 540)
        #else
        min(size.width * 9 / 16, size.height * 0.48)
        #endif
    }

    private func togglePlayerSize() {
        #if os(macOS)
        let window = NSApp.keyWindow
        let windowIsFullScreen = window?.styleMask.contains(.fullScreen) == true
        if model.isPlayerExpanded {
            if windowIsFullScreen {
                window?.toggleFullScreen(nil)
            }
        } else if !windowIsFullScreen {
            window?.toggleFullScreen(nil)
        }
        #endif
        withAnimation(.easeInOut(duration: 0.24)) {
            model.isPlayerExpanded.toggle()
        }
    }
}
