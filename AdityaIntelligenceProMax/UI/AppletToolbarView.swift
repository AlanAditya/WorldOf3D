//
//  AppletToolbarView.swift
//  AdityaIntelligenceProMax
//
//  Draws Intelligence.appletToolbar (UI/AppletToolbar.h) as a small floating capsule: Liquid
//  Glass on macOS 26 / iOS 26, a thin material before. 3D_Editor.swift puts it at the bottom
//  right of the renderer.
//

import SwiftUI

// Mirrors the Objective-C toolbar's items so SwiftUI redraws when an applet adds/removes one
// or a toggle flips.
final class AppletToolbarModel: ObservableObject {
    let toolbar: AppletToolbar
    @Published private(set) var items: [AppletToolbarItem] = []
    @Published private(set) var legend: [AppletLegendEntry] = []
    @Published private(set) var revision = 0   // bumps on toggle changes (same items, new state)

    init(toolbar: AppletToolbar) {
        self.toolbar = toolbar
        items = toolbar.items
        legend = toolbar.legend
        toolbar.onItemsChanged = { [weak self] in
            guard let self else { return }
            self.items = self.toolbar.items
            self.legend = self.toolbar.legend
            self.revision += 1
        }
    }
}

struct AppletToolbarView: View {
    @StateObject private var model: AppletToolbarModel

    init(toolbar: AppletToolbar) {
        _model = StateObject(wrappedValue: AppletToolbarModel(toolbar: toolbar))
    }

    var body: some View {
        // The legend card sits right above the capsule, right-aligned with it.
        VStack(alignment: .trailing, spacing: 8) {
            if !model.legend.isEmpty {
                LegendCard(entries: model.legend)
            }
            if !model.items.isEmpty {
                HStack(spacing: 2) {
                    ForEach(Array(model.items.enumerated()), id: \.offset) { _, item in
                        control(for: item)
                    }
                }
                .padding(.horizontal, 6)
                .padding(.vertical, 4)
                .modifier(CapsuleGlass())
            }
        }
    }

    @ViewBuilder
    private func control(for item: AppletToolbarItem) -> some View {
        switch item.kind {
        case .toggle:
            Button {
                model.toolbar.set(item, on: !item.on)
            } label: {
                ToolbarIcon(symbol: item.icon, highlighted: item.on)
            }
            .buttonStyle(.plain)
            .help(item.title)
            .accessibilityLabel(item.title)
            .accessibilityAddTraits(item.on ? .isSelected : [])
        case .menu:
            Menu {
                Section(item.title) {
                    ForEach(Array(item.menuTitles.enumerated()), id: \.offset) { index, title in
                        let icon = item.menuIcons[index]
                        Button {
                            model.toolbar.select(item, index: index)
                        } label: {
                            if icon.isEmpty { Text(title) } else { Label(title, systemImage: icon) }
                        }
                    }
                }
            } label: {
                ToolbarIcon(symbol: item.icon, highlighted: false)
            }
            .menuStyle(.button)
            .buttonStyle(.plain)
            .menuIndicator(.hidden)
            .fixedSize()
            .help(item.title)
            .accessibilityLabel(item.title)
        default:
            Button {
                model.toolbar.press(item)
            } label: {
                ToolbarIcon(symbol: item.icon, highlighted: false)
            }
            .buttonStyle(.plain)
            .help(item.title)
            .accessibilityLabel(item.title)
        }
    }
}

private struct ToolbarIcon: View {
    let symbol: String
    let highlighted: Bool
#if os(iOS)
    private let size: CGFloat = 40   // comfortable touch target
#else
    private let size: CGFloat = 30
#endif

    var body: some View {
        Image(systemName: symbol)
            .font(.system(size: size * 0.45, weight: .medium))
            .foregroundStyle(highlighted ? Color.accentColor : Color.primary)
            .frame(width: size, height: size)
            .background {
                if highlighted {
                    Circle().fill(Color.accentColor.opacity(0.18))
                }
            }
            .contentShape(Circle())
    }
}

// The legend (AppletToolbar's setLegend:): sections of colour swatches with a label each, on a
// translucent card. It only informs, so it lets clicks and drags through to the renderer.
private struct LegendCard: View {
    let entries: [AppletLegendEntry]

    // A section header and the lines under it.
    private struct Block {
        var title: String?
        var lines: [AppletLegendEntry] = []
    }

    private var blocks: [Block] {
        var out: [Block] = []
        for e in entries {
            if e.kind == .section || out.isEmpty { out.append(Block(title: e.kind == .section ? e.label : nil)) }
            if e.kind != .section { out[out.count - 1].lines.append(e) }
        }
        return out
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                VStack(alignment: .leading, spacing: 4) {
                    if let title = block.title {
                        Text(title.uppercased())
                            .font(.system(size: 9, weight: .semibold))
                            .tracking(0.6)
                            .foregroundStyle(.secondary)
                    }
                    // Long lists (e.g. all 20 classes) in two columns.
                    if block.lines.count > 8 {
                        LazyVGrid(columns: [GridItem(.flexible(), alignment: .leading),
                                            GridItem(.flexible(), alignment: .leading)],
                                  alignment: .leading, spacing: 4) {
                            ForEach(Array(block.lines.enumerated()), id: \.offset) { _, e in LegendLine(entry: e) }
                        }
                    } else {
                        ForEach(Array(block.lines.enumerated()), id: \.offset) { _, e in LegendLine(entry: e) }
                    }
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .frame(maxWidth: 300, alignment: .leading)
        .fixedSize(horizontal: false, vertical: true)
        .modifier(CardGlass())
        .allowsHitTesting(false)
    }
}

private struct LegendLine: View {
    let entry: AppletLegendEntry

    // The renderer's values are linear (the sRGB drawable encodes them), so match that.
    private var color: Color {
        Color(.sRGBLinear, red: Double(entry.red), green: Double(entry.green), blue: Double(entry.blue))
    }

    var body: some View {
        switch entry.kind {
        case .note:
            Text(entry.label)
                .font(.system(size: 10))
                .foregroundStyle(.secondary)
        default:
            HStack(spacing: 7) {
                Group {
                    if entry.kind == .outline {
                        RoundedRectangle(cornerRadius: 2)
                            .strokeBorder(color, lineWidth: 1.5)
                            .shadow(color: color.opacity(0.8), radius: 2)
                    } else {
                        RoundedRectangle(cornerRadius: 3).fill(color)
                    }
                }
                .frame(width: 12, height: 12)
                Text(entry.label)
                    .font(.system(size: 11))
                    .lineLimit(1)
            }
        }
    }
}

// Liquid Glass card on macOS 26 / iOS 26, a frosted card before.
private struct CardGlass: ViewModifier {
    func body(content: Content) -> some View {
        if #available(macOS 26.0, iOS 26.0, *) {
            content.glassEffect(.regular, in: .rect(cornerRadius: 14))
        } else {
            content
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
                .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(.white.opacity(0.15)))
                .shadow(color: .black.opacity(0.2), radius: 8, y: 2)
        }
    }
}

// Liquid Glass capsule on macOS 26 / iOS 26, a frosted capsule before.
private struct CapsuleGlass: ViewModifier {
    func body(content: Content) -> some View {
        if #available(macOS 26.0, iOS 26.0, *) {
            GlassEffectContainer {
                content.glassEffect(.regular.interactive(), in: .capsule)
            }
        } else {
            content
                .background(.ultraThinMaterial, in: Capsule())
                .overlay(Capsule().strokeBorder(.white.opacity(0.15)))
                .shadow(color: .black.opacity(0.2), radius: 8, y: 2)
        }
    }
}

extension View {
    // Pins the capsule to the bottom-right corner of this view (the renderer). `bottomInset` is
    // measured from the view's real bottom edge, safe area ignored, so iOS can lift it above
    // the floating sheet's smallest detent.
    func appletToolbar(_ toolbar: AppletToolbar, bottomInset: CGFloat = 12) -> some View {
        overlay {
            AppletToolbarView(toolbar: toolbar)
                .padding(.trailing, 16)
                .padding(.bottom, bottomInset)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomTrailing)
                .ignoresSafeArea()
        }
    }
}
