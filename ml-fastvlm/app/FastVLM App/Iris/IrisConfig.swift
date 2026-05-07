//
// IrisConfig.swift
// One place for every tunable in the Iris OS layer. The defaults mirror
// sentinel.py and inference.py from the desktop project so behaviour
// matches between platforms.
//

import Foundation

enum IrisConfig {
    // MARK: - Sentinel (camera Tier 1)

    /// Frames kept in the ring buffer. At ~30 fps this is ~0.5 s of memory.
    static let bufferSize: Int = 15

    /// Laplacian variance below this is "blurry" — we still pass the frame
    /// through but flag it. Mirrors sentinel.py blur_threshold.
    static let blurThreshold: Float = 100.0

    /// Average motion below this over `stableHistory` frames means the
    /// camera is stable enough to read. Tuned for handheld phones.
    static let maxMotion: Float = 4.5

    /// Number of trailing frames inspected for stability.
    static let stableHistory: Int = 8

    /// A spike above this in any recent frame counts as "new content".
    static let baselineMotion: Float = 8.0

    // MARK: - Cognition (Tier 2)

    /// JSON-output prompt cap. The Python desktop uses 100 tokens; the
    /// stock Apple app used 240. We want fast TTFT, not paragraphs.
    static let maxTokens: Int = 100

    /// Greedy decoding. Don't let a 1.5B model invent extra fields.
    static let temperature: Float = 0.0

    /// Friendly model name shown in the "Source:" badge.
    static let modelDisplayName: String = "FastVLM-1.5B"

    // MARK: - Cooldown

    /// How long to leave the answer on screen before re-arming Sentinel.
    static let cooldownSeconds: Double = 5.0

    // MARK: - Tavily

    /// Read once at app launch from Info.plist (key injected by fastlane
    /// from $TAVILY_API_KEY). Empty / nil = web_search falls back to
    /// Wikipedia only.
    static var tavilyAPIKey: String? {
        let raw = Bundle.main.object(forInfoDictionaryKey: "TavilyAPIKey") as? String
        guard let raw, !raw.isEmpty, raw != "$(TAVILY_API_KEY)" else { return nil }
        return raw
    }
}
