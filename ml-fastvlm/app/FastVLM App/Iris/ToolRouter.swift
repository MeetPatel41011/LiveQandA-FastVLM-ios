//
// ToolRouter.swift
// Deterministic Python-side router from inference.py:170-189.
//
// Priority: matrix > scalar math > web search > raw LLM answer.
// We match against `text_in_image` (the OCR'd question) rather than the
// LLM's `answer`, because small models often get the question right
// even when their direct answer is wrong.
//

import Foundation

struct RoutedAnswer {
    let body: String
    let source: String
}

enum ToolRouter {

    // Same patterns as inference.py.
    private static let mathPattern: NSRegularExpression = {
        // 45 * 12, 100/4+3, etc. (`=?` is optional trailing equals)
        let p = #"\d+\s*[\+\-\*\/\^\%]\s*\d+(?:\s*=\s*[\?]?)?"#
        return try! NSRegularExpression(pattern: p, options: [.caseInsensitive])
    }()

    private static let matrixPattern: NSRegularExpression = {
        let p = #"\[|matrix\s*multiply|C\s*=\s*AB|A\s*=|B\s*="#
        return try! NSRegularExpression(pattern: p, options: [.caseInsensitive])
    }()

    private static let webPattern: NSRegularExpression = {
        let p = #"news|search|latest|current|price|who is|what is|where is|when was|today|yesterday|now"#
        return try! NSRegularExpression(pattern: p, options: [.caseInsensitive])
    }()

    /// Pick a tool, run it, return a stamped answer.
    /// `directAnswer` is the model's own answer text — used only when no
    /// tool matches.
    static func route(
        textInImage: String,
        directAnswer: String,
        rawJSON: String
    ) async -> RoutedAnswer {
        let model = IrisConfig.modelDisplayName

        if matches(matrixPattern, in: textInImage) {
            // Matrix tool wants the full blob (numbers can be in either field).
            let body = Tools.matrixMultiply(rawJSON + " " + directAnswer)
            return RoutedAnswer(body: body, source: "LLM (\(model)) + Tool (matrix)")
        }

        if let mathExpr = firstMatch(mathPattern, in: textInImage) {
            // Strip trailing `=` / `=?` like inference.py does.
            let cleaned = mathExpr
                .trimmingCharacters(in: .whitespaces)
                .trimmingCharacters(in: CharacterSet(charactersIn: "=? "))
            let body = Tools.calculator(cleaned)
            return RoutedAnswer(body: body, source: "LLM (\(model)) + Tool (calculator)")
        }

        if matches(webPattern, in: textInImage) {
            let body = await Tools.webSearch(textInImage)
            return RoutedAnswer(body: body, source: "LLM (\(model)) + Tool (web_search)")
        }

        // Fall back to the model's own answer.
        let body = directAnswer.isEmpty
            ? "(no answer extracted)"
            : directAnswer
        return RoutedAnswer(body: body, source: "LLM (\(model))")
    }

    // MARK: - Helpers

    private static func matches(_ regex: NSRegularExpression, in s: String) -> Bool {
        let r = NSRange(s.startIndex..<s.endIndex, in: s)
        return regex.firstMatch(in: s, range: r) != nil
    }

    private static func firstMatch(_ regex: NSRegularExpression, in s: String) -> String? {
        let r = NSRange(s.startIndex..<s.endIndex, in: s)
        guard let m = regex.firstMatch(in: s, range: r),
              let mr = Range(m.range, in: s) else { return nil }
        return String(s[mr])
    }
}
