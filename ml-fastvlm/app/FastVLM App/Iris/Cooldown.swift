//
// Cooldown.swift
// Simple monotonic-time gate that prevents the Sentinel auto-fire loop
// from re-triggering immediately after an answer is shown. Mirrors the
// `last_inference_time + cooldown_seconds` check in main.py.
//

import Foundation

@MainActor
final class Cooldown {
    private var armedAt: Date = .distantPast
    private(set) var duration: Double

    init(duration: Double = IrisConfig.cooldownSeconds) {
        self.duration = duration
    }

    /// True while the cooldown window is still open.
    var isActive: Bool {
        Date().timeIntervalSince(armedAt) < duration
    }

    /// Seconds remaining (clamped to 0). Used by PerfHUD.
    var remaining: Double {
        max(0, duration - Date().timeIntervalSince(armedAt))
    }

    /// Start the cooldown. Idempotent; calling again restarts the timer.
    func start(_ secs: Double? = nil) {
        if let s = secs { duration = s }
        armedAt = Date()
    }

    /// Cancel the cooldown immediately (used for manual modes).
    func clear() {
        armedAt = .distantPast
    }
}
