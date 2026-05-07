//
// IrisAgent.swift
// The Swift mirror of inference.py + main.py: orchestrates Sentinel,
// PromptBuilder, FastVLMModel, JSONResponseParser, and ToolRouter into
// one async pipeline. ContentView treats this as a single observable
// box.
//

import CoreImage
import CoreVideo
import Foundation
import MLXLMCommon

@Observable
@MainActor
final class IrisAgent {

    // MARK: - State surfaced to the UI

    var lastAnswer: RoutedAnswer?

    /// Mirror of FastVLMModel.lastTTFTms for the PerfHUD overlay.
    var lastTTFTms: Int = 0

    /// True while a generate call is in flight.
    var isThinking: Bool = false

    /// Latest error surfaced to the UI (nil when healthy).
    var errorMessage: String?

    let cooldown = Cooldown()

    // MARK: - Collaborators

    private let model: FastVLMModel

    init(model: FastVLMModel) {
        self.model = model
    }

    // MARK: - Public API

    /// Run one Iris cycle: build prompt, call model, parse JSON, route to
    /// a tool, store result. Caller (ContentView) decides WHEN to invoke
    /// (Sentinel-gated auto, or manual button). Idempotent guard against
    /// concurrent calls — second concurrent call short-circuits.
    func ask(frame: CVPixelBuffer, spokenTranscript: String?) async {
        guard !isThinking else { return }
        isThinking = true
        errorMessage = nil
        defer { isThinking = false }

        let prompt = PromptBuilder.buildPrompt(spokenTranscript: spokenTranscript)
        let userInput = UserInput(
            prompt: .text(prompt),
            images: [.ciImage(CIImage(cvPixelBuffer: frame))]
        )

        do {
            let raw = try await model.generateForIris(userInput)
            lastTTFTms = model.lastTTFTms

            let parsed = JSONResponseParser.parse(raw)
            let routed = await ToolRouter.route(
                textInImage: parsed.textInImage,
                directAnswer: parsed.answer,
                rawJSON: raw
            )
            lastAnswer = routed
            cooldown.start()
        } catch {
            errorMessage = "Iris failed: \(error.localizedDescription)"
        }
    }

    /// Clear the displayed answer (e.g. when user toggles modes).
    func reset() {
        lastAnswer = nil
        cooldown.clear()
    }
}
