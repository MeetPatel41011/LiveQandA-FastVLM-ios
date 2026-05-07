//
// PromptBuilder.swift
// Builds the JSON-guided prompt that turns a 1.5B chat model into a
// reliable structured-output extractor. Mirrors the rule_instruction
// + prefilled `{"text_in_image": "` trick from inference.py.
//

import Foundation

enum PromptBuilder {

    /// The instruction text. Identical wording to inference.py:108-117 so
    /// behaviour is comparable across platforms.
    private static let ruleInstruction = """
    Analyze the image and extract the question written in it.
    You MUST output a valid JSON object with exactly two fields.
    Format: {"text_in_image": "<question text>", "answer": "<factual answer>"}
    """

    /// What the user-visible "task" line says. The Python project uses
    /// the same string in main.py.
    static let strictTaskInstruction =
        "Read the text in the image. Write the factual answer to that question. Do NOT transcribe the question itself."

    /// Build the user-facing prompt string.
    ///
    /// - Parameter spokenTranscript: optional speech transcript captured by
    ///   SpeechController. When non-empty it's surfaced as a separate
    ///   "Spoken question:" line so the VLM has both modalities.
    /// - Returns: a single string ready to hand to UserInput(prompt: .text(...)).
    ///
    /// On prefix-fill: MLXLMCommon's UserInput doesn't expose an
    /// "assistant-already-said-this" hook the way HuggingFace generate()
    /// does. Option A (this implementation) embeds the prefill literal at
    /// the *end of the user message*. With temperature=0.0 + greedy decode
    /// the assistant overwhelmingly continues from where we left off.
    /// If we ever see JSON quality regress, switch to Option B (subclass
    /// UserInputProcessor and inject prefill tokens after templating).
    static func buildPrompt(spokenTranscript: String? = nil) -> String {
        var parts: [String] = [ruleInstruction]

        if let t = spokenTranscript?.trimmingCharacters(in: .whitespacesAndNewlines),
           !t.isEmpty {
            parts.append("Spoken question: \(t)")
        }

        parts.append(strictTaskInstruction)

        // The prefill: by ending the user turn with a partial JSON the
        // assistant has nowhere to go but continue the JSON.
        parts.append(#"Begin your reply with: {"text_in_image": ""#)

        return parts.joined(separator: "\n\n")
    }
}
