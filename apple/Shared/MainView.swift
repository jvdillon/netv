import SwiftUI

struct MainView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        #if os(tvOS)
        GuideMainView()
            .ignoresSafeArea(.container, edges: model.isPlayerExpanded ? .all : [])
        #elseif os(macOS)
        GuideMainView()
        #else
        TabView {
            NavigationStack {
                WatchView()
            }
            .tabItem { Label("Live", systemImage: "dot.radiowaves.left.and.right") }

            NavigationStack {
                SettingsView()
            }
            .tabItem { Label("Settings", systemImage: "gearshape.fill") }
        }
        #endif
    }
}

#if os(macOS) || os(tvOS)
private enum GuideDestination: String, CaseIterable, Identifiable {
    case live
    case settings

    var id: Self { self }
    var title: String { self == .live ? "Live TV" : "Settings" }
    var icon: String { self == .live ? "dot.radiowaves.left.and.right" : "gearshape.fill" }
}

private struct GuideMainView: View {
    @EnvironmentObject private var model: AppModel
    @State private var destination: GuideDestination = .live
    @FocusState private var focusedControl: String?
    #if os(tvOS)
    @State private var isSearchPresented = false
    #endif

    var body: some View {
        HStack(spacing: 0) {
            // Keep the content's position in the view hierarchy stable while
            // fullscreen hides the rail, preserving the mounted player.
            navigationRail
                .frame(width: model.isPlayerExpanded ? 0 : GuideMetrics.scaled(64))
                .clipped()
                .opacity(model.isPlayerExpanded ? 0 : 1)
                .allowsHitTesting(!model.isPlayerExpanded)
                .disabled(model.isPlayerExpanded)
                .accessibilityHidden(model.isPlayerExpanded)
            Group {
                switch destination {
                case .live:
                    WatchView()
                case .settings:
                    SettingsView()
                }
            }
        }
        .background(GuideTheme.background.ignoresSafeArea())
        .tint(GuideTheme.program)
        #if os(macOS)
        .toolbar {
            // macOS 26 wraps toolbar items in a glass pill; the title is plain text.
            if #available(macOS 26.0, *) {
                ToolbarItem(placement: .principal) {
                    Text(destination.title)
                        .font(.headline)
                }
                .sharedBackgroundVisibility(.hidden)
            } else {
                ToolbarItem(placement: .principal) {
                    Text(destination.title)
                        .font(.headline)
                }
            }
            if destination == .live {
                ToolbarItem(placement: .automatic) {
                    TextField("Search channels and programs", text: $model.query)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 260)
                }
                ToolbarItem(placement: .automatic) {
                    Button {
                        Task { await model.loadGuide() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .disabled(model.isLoading)
                    .help("Refresh guide")
                    .accessibilityLabel("Refresh guide")
                }
            }
        }
        .toolbar(model.isPlayerExpanded ? .hidden : .automatic)
        #else
        .sheet(isPresented: $isSearchPresented) {
            VStack(alignment: .leading, spacing: 30) {
                Text("Search Live TV")
                    .font(.title.bold())
                TextField("Channels and programs", text: $model.query)
                Button("Done") { isSearchPresented = false }
                    .buttonStyle(.borderedProminent)
            }
            .padding(60)
            .background(GuideTheme.background)
            .onExitCommand { isSearchPresented = false }
        }
        // With no in-app destination to dismiss, leave Back unhandled so
        // tvOS can return to the Apple TV Home screen.
        .onExitCommand(perform: canHandleExitCommand ? handleExitCommand : nil)
        .onPlayPauseCommand {
            if destination == .live {
                model.playPauseRequest = UUID()
            }
        }
        #endif
    }

    private var navigationRail: some View {
        VStack(spacing: GuideMetrics.scaled(18)) {
            ForEach(GuideDestination.allCases) { item in
                if item == .settings {
                    #if os(tvOS)
                    railButton("Search", icon: "magnifyingglass", id: "search") {
                        isSearchPresented = true
                    }
                    railButton("Refresh guide", icon: "arrow.clockwise", id: "refresh") {
                        Task { await model.loadGuide() }
                    }
                    .disabled(model.isLoading)
                    #endif
                    Spacer()
                }
                railButton(item.title, icon: item.icon, id: item.rawValue, isSelected: destination == item) {
                    destination = item
                }
            }
        }
        .padding(.vertical, GuideMetrics.scaled(22))
        .frame(width: GuideMetrics.scaled(64))
        .frame(maxHeight: .infinity)
        .background(GuideTheme.background)
        .overlay(alignment: .trailing) {
            GuideTheme.divider.frame(width: 1)
        }
        .focusSection()
    }

    private func railButton(
        _ title: String, icon: String, id: String,
        isSelected: Bool = false, action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.system(size: GuideMetrics.fontSize(18), weight: .medium))
                .frame(width: GuideMetrics.scaled(42), height: GuideMetrics.scaled(44))
        }
        .buttonStyle(GuideButtonStyle(isSelected: isSelected, isFocused: focusedControl == id))
        .focused($focusedControl, equals: id)
        #if os(macOS)
        .help(title)
        #endif
        .accessibilityLabel(title)
        .accessibilityValue(isSelected ? "Selected" : "")
    }

    #if os(tvOS)
    private var canHandleExitCommand: Bool {
        model.isPlayerExpanded || destination != .live
    }

    private func handleExitCommand() {
        if model.isPlayerExpanded {
            model.isPlayerExpanded = false
        } else {
            destination = .live
            focusedControl = GuideDestination.live.rawValue
        }
    }
    #endif
}
#endif
