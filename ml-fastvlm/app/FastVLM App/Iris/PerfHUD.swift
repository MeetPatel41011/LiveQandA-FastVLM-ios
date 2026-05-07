//
// PerfHUD.swift
// Toggleable overlay that surfaces the four numbers we care about while
// tuning Iris on a real iPhone:
//   - TTFT (ms) for the most recent inference
//   - Sentinel state (sharpness + motion)
//   - Cooldown countdown
//   - Whether Sentinel currently considers the scene stable & new
//
// Show via a debug build flag or a long-press gesture in ContentView.
//

import SwiftUI

struct PerfHUD: View {

    let ttftMs: Int
    let sharpness: Float
    let motion: Float
    let isStable: Bool
    let hasNewContent: Bool
    let cooldownRemaining: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            row(label: "TTFT", value: ttftMs > 0 ? "\(ttftMs) ms" : "—")
            row(label: "Sharp", value: String(format: "%.0f", sharpness))
            row(label: "Motion", value: String(format: "%.1f", motion))
            row(label: "Stable", value: isStable ? "yes" : "no")
            row(label: "New", value: hasNewContent ? "yes" : "no")
            row(label: "Cooldown", value: cooldownRemaining > 0
                ? String(format: "%.1fs", cooldownRemaining) : "ready")
        }
        .font(.system(size: 11, weight: .regular, design: .monospaced))
        .foregroundStyle(.white)
        .padding(8)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.black.opacity(0.55))
        )
    }

    @ViewBuilder
    private func row(label: String, value: String) -> some View {
        HStack(spacing: 8) {
            Text(label)
                .frame(width: 64, alignment: .leading)
                .foregroundStyle(.white.opacity(0.7))
            Text(value)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}
