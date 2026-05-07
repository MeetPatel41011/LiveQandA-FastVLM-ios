//
// JSONResponseParser.swift
// Tolerant parser for the {"text_in_image": "...", "answer": "..."}
// blob produced by the JSON-guided prompt. Mirrors inference.py:160-168.
//
// Why tolerant: a 1.5B model — even with greedy decoding — sometimes
// drops the closing quote/brace, or runs over the max-tokens limit
// mid-string. We:
//   1. Append `"}` if the string doesn't end with `}`.
//   2. Try JSONDecoder.
//   3. On failure, fall back to two regex captures.
//

import Foundation

struct IrisAnswer: Codable, Equatable {
    let textInImage: String
    let answer: String

    enum CodingKeys: String, CodingKey {
        case textInImage = "text_in_image"
        case answer
    }
}

enum JSONResponseParser {

    /// Parse what the model produced. The input string is whatever the
    /// MLX streamer accumulated; we don't need to reattach the prefilled
    /// `{"text_in_image": "` prefix — Option A in PromptBuilder asks the
    /// model to write it itself.
    static func parse(_ raw: String) -> IrisAnswer {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalised = repaired(trimmed)

        if let data = normalised.data(using: .utf8),
           let parsed = try? JSONDecoder().decode(IrisAnswer.self, from: data) {
            return parsed
        }

        // Regex fallback — same patterns as inference.py.
        let textInImage = extract(field: "text_in_image", from: normalised) ?? ""
        let answer      = extract(field: "answer",        from: normalised) ?? ""
        return IrisAnswer(textInImage: textInImage, answer: answer)
    }

    // MARK: - Repair

    private static func repaired(_ raw: String) -> String {
        // If the model already produced a valid-looking JSON object, we
        // leave it alone. Otherwise we append the smallest suffix that
        // makes it parseable.
        if raw.hasPrefix("{") && raw.hasSuffix("}") {
            return raw
        }
        if !raw.hasPrefix("{") {
            // Sometimes the model emits chat-style preamble. Try to find
            // the first `{` and crop.
            if let openIdx = raw.firstIndex(of: "{") {
                return repaired(String(raw[openIdx...]))
            }
            return "{\"text_in_image\": \"\", \"answer\": \"\(raw.replacingOccurrences(of: "\"", with: "\\\""))\"}"
        }
        // Starts with `{` but no closing brace.
        let needsQuote = !raw.hasSuffix("\"")
        return raw + (needsQuote ? "\"}" : "}")
    }

    // MARK: - Regex fallback

    private static func extract(field: String, from raw: String) -> String? {
        let pattern = "\"\(NSRegularExpression.escapedPattern(for: field))\"\\s*:\\s*\"((?:\\\\.|[^\"\\\\])*)\""
        guard let regex = try? NSRegularExpression(
            pattern: pattern,
            options: [.dotMatchesLineSeparators]
        ) else { return nil }

        let range = NSRange(raw.startIndex..<raw.endIndex, in: raw)
        guard let match = regex.firstMatch(in: raw, options: [], range: range),
              match.numberOfRanges >= 2,
              let captureRange = Range(match.range(at: 1), in: raw)
        else { return nil }

        // Decode common JSON escapes \" and \\.
        let value = String(raw[captureRange])
            .replacingOccurrences(of: "\\\"", with: "\"")
            .replacingOccurrences(of: "\\\\", with: "\\")
            .replacingOccurrences(of: "\\n",  with: "\n")
        return value.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
