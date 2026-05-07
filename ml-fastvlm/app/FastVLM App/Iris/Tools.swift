//
// Tools.swift
// Swift port of tools.py — calculator, complex matrix multiplier, and
// multi-source web search. Each function is `async throws` and returns
// a plain String the UI can render directly.
//

import Foundation

enum IrisToolError: Error, LocalizedError {
    case noMathExpression
    case evalFailure(String)
    case matrixParseFailure(String)
    case networkFailure(String)

    var errorDescription: String? {
        switch self {
        case .noMathExpression:        return "No math expression found."
        case .evalFailure(let s):      return "Eval error: \(s)"
        case .matrixParseFailure(let s): return "Matrix parse error: \(s)"
        case .networkFailure(let s):   return "Network error: \(s)"
        }
    }
}

enum Tools {

    // ─────────────────────────────────────────────────────────────
    // Tool 1: Scalar calculator (NSExpression — sandboxed, no `eval`).
    // ─────────────────────────────────────────────────────────────

    /// Evaluate `45 * 12`, `100 / 4 + 3`, etc. Strips everything outside
    /// `[0-9 + - * / . ( )]` first to keep NSExpression away from any
    /// keypath-style hostile input.
    static func calculator(_ raw: String) -> String {
        let safeChars = CharacterSet(charactersIn: "0123456789+-*/.() ")
        let safe = raw.unicodeScalars.filter { safeChars.contains($0) }
        let expr = String(String.UnicodeScalarView(safe))
            .trimmingCharacters(in: .whitespaces)

        guard !expr.isEmpty else {
            return "Error: No math expression found in '\(raw)'"
        }

        let nsExpr = NSExpression(format: expr)
        guard let result = nsExpr.expressionValue(with: nil, context: nil) else {
            return "Error evaluating '\(expr)'"
        }
        // NSExpression returns NSNumber for numerics. Format ints cleanly.
        if let n = result as? NSNumber {
            let d = n.doubleValue
            if d == d.rounded() && abs(d) < 1e15 {
                return String(Int64(d))
            }
            return String(d)
        }
        return "\(result)"
    }

    // ─────────────────────────────────────────────────────────────
    // Tool 2: Complex matrix multiplier.
    // ─────────────────────────────────────────────────────────────

    /// Parse all numbers (real + complex) out of `problemText`, split into
    /// two equal-size square matrices, multiply, and pretty-print.
    /// Mirrors execute_matrix_multiply in tools.py.
    static func matrixMultiply(_ problemText: String) -> String {
        // Numbers: optional sign, digits, optional decimal, optional `i`.
        // Pulls "1+i", "4-i", "2", "i" etc.
        let numberPattern = #"(?<![A-Za-z])(?:[+\-]?\s*\d+(?:\.\d+)?\s*[+\-]\s*\d*\s*i|[+\-]?\s*\d+(?:\.\d+)?|[+\-]?\s*\d*\s*i)(?![A-Za-z])"#

        guard let regex = try? NSRegularExpression(pattern: numberPattern) else {
            return "Error: matrix regex failed."
        }
        let nsRange = NSRange(problemText.startIndex..<problemText.endIndex, in: problemText)
        let matches = regex.matches(in: problemText, range: nsRange)

        let raw: [String] = matches.compactMap {
            guard let r = Range($0.range, in: problemText) else { return nil }
            let s = String(problemText[r]).trimmingCharacters(in: .whitespaces)
            return s.isEmpty || s == "+" || s == "-" ? nil : s
        }
        let parsed: [Complex] = raw.map { parseComplex($0) }

        guard parsed.count >= 4 else {
            return "Error: Could not extract enough numbers (got \(parsed.count))."
        }

        // Half goes to A, half to B.
        let half = parsed.count / 2
        let aNums = Array(parsed.prefix(half))
        let bNums = Array(parsed.suffix(parsed.count - half))

        let size = Int(Double(aNums.count).squareRoot().rounded())
        guard size * size == aNums.count, size * size == bNums.count else {
            return "Error: Matrix is not square (found \(aNums.count) elements per matrix)."
        }

        var c = [[Complex]](repeating: [Complex](repeating: .zero, count: size), count: size)
        for i in 0..<size {
            for j in 0..<size {
                var s = Complex.zero
                for k in 0..<size {
                    s = s + aNums[i * size + k] * bNums[k * size + j]
                }
                c[i][j] = s
            }
        }

        var lines: [String] = ["C ="]
        for row in c {
            let cells = row.map { format($0) }.joined(separator: ",  ")
            lines.append("  [\(cells)]")
        }
        return lines.joined(separator: "\n")
    }

    // ─────────────────────────────────────────────────────────────
    // Tool 3: Web search (Wikipedia → Tavily fallback).
    // ─────────────────────────────────────────────────────────────

    /// Multi-source factual lookup. Wikipedia first (free, fast,
    /// definition-quality). Tavily second (LLM-curated snippets, requires
    /// `TavilyAPIKey` in Info.plist).
    ///
    /// DuckDuckGo (the third source in tools.py) is intentionally omitted:
    /// its iOS-friendly REST surface is unstable and Tavily is strictly
    /// better when the key is present.
    static func webSearch(_ query: String) async -> String {
        let cleaned = cleanQuery(query)
        guard !cleaned.isEmpty else { return "Error: empty search query." }

        let isDefinition = ["what is", "who is", "define", "meaning of"]
            .contains { query.lowercased().contains($0) }

        // 1. Wikipedia.
        if isDefinition,
           let wiki = await wikipediaSummary(for: cleaned) {
            return "[Wikipedia] \(wiki)"
        }

        // 2. Tavily.
        if let key = IrisConfig.tavilyAPIKey,
           let tav = await tavilySearch(query: query, key: key) {
            return "[Tavily] \(tav)"
        }

        // Last-resort Wikipedia even for non-definition queries.
        if let wiki = await wikipediaSummary(for: cleaned) {
            return "[Wikipedia] \(wiki)"
        }

        return "Sorry, I couldn't find a specific answer for '\(cleaned)'."
    }

    // MARK: - Web helpers

    private static func cleanQuery(_ raw: String) -> String {
        // Strip embedded JSON noise.
        var s = raw
        if let range = s.range(of: #"\{.*?\}"#, options: .regularExpression) {
            s.removeSubrange(range)
        }
        // Strip leading interrogatives.
        let prefixes = ["what is ", "who is ", "where is ", "when was ",
                        "tell me about ", "search for ", "how much is ",
                        "current price of ", "define "]
        let lower = s.lowercased()
        for p in prefixes where lower.hasPrefix(p) {
            s = String(s.dropFirst(p.count))
            break
        }
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "?"))
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func wikipediaSummary(for query: String) async -> String? {
        let slug = query.replacingOccurrences(of: " ", with: "_")
        guard let encoded = slug.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed),
              let url = URL(string: "https://en.wikipedia.org/api/rest_v1/page/summary/\(encoded)")
        else { return nil }

        var req = URLRequest(url: url, timeoutInterval: 5)
        req.setValue("Iris/1.0 (iOS)", forHTTPHeaderField: "User-Agent")

        do {
            let (data, response) = try await URLSession.shared.data(for: req)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200,
                  let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let extract = json["extract"] as? String,
                  !extract.isEmpty
            else { return nil }
            return extract
        } catch {
            return nil
        }
    }

    private static func tavilySearch(query: String, key: String) async -> String? {
        guard let url = URL(string: "https://api.tavily.com/search") else { return nil }
        var req = URLRequest(url: url, timeoutInterval: 8)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body: [String: Any] = [
            "api_key": key,
            "query": query,
            "search_depth": "basic",
            "max_results": 1
        ]
        req.httpBody = try? JSONSerialization.data(withJSONObject: body)

        do {
            let (data, response) = try await URLSession.shared.data(for: req)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200,
                  let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let results = json["results"] as? [[String: Any]],
                  let first = results.first,
                  let content = first["content"] as? String,
                  !content.isEmpty
            else { return nil }
            return content
        } catch {
            return nil
        }
    }

    // MARK: - Complex math (no swift-numerics dep)

    struct Complex: Equatable {
        var real: Double
        var imag: Double

        static let zero = Complex(real: 0, imag: 0)

        static func + (a: Complex, b: Complex) -> Complex {
            Complex(real: a.real + b.real, imag: a.imag + b.imag)
        }
        static func * (a: Complex, b: Complex) -> Complex {
            Complex(
                real: a.real * b.real - a.imag * b.imag,
                imag: a.real * b.imag + a.imag * b.real
            )
        }
    }

    private static func format(_ c: Complex) -> String {
        let r = roundClean(c.real)
        let i = roundClean(c.imag)
        if i == 0 { return formatNumber(r) }
        let sign = i >= 0 ? "+" : "-"
        return "\(formatNumber(r)) \(sign) \(formatNumber(abs(i)))i"
    }

    private static func roundClean(_ v: Double) -> Double {
        let r = (v * 10000).rounded() / 10000
        return r == 0 ? 0 : r
    }

    private static func formatNumber(_ v: Double) -> String {
        if v == v.rounded() { return String(Int64(v)) }
        return String(v)
    }

    /// Parse `1+i`, `4-i`, `2`, `i`, `3.5`, `-2i`, etc.
    private static func parseComplex(_ raw: String) -> Complex {
        var s = raw.replacingOccurrences(of: " ", with: "")
        if s.isEmpty { return .zero }

        // Plain number, no `i` at all.
        if !s.contains("i") {
            return Complex(real: Double(s) ?? 0, imag: 0)
        }

        // Pure imaginary (no real part).
        // Match "i", "+i", "-i", "2i", "-3i", "1.5i".
        if !s.contains("+") || s.firstIndex(of: "+") == s.startIndex {
            // No real part: try to extract leading sign + magnitude.
            return Complex(real: 0, imag: parseImagMagnitude(s))
        }
        if s.dropFirst().contains("-") == false && s.contains("i") && !s.contains("+") {
            return Complex(real: 0, imag: parseImagMagnitude(s))
        }

        // Real ± imag form, e.g. "1+i", "4-2i", "-3.5+i".
        // Find the operator that separates the two parts. Skip a leading
        // sign on position 0.
        var splitIdx: String.Index?
        var idx = s.index(after: s.startIndex)
        while idx < s.endIndex {
            let ch = s[idx]
            if ch == "+" || ch == "-" {
                splitIdx = idx
                break
            }
            idx = s.index(after: idx)
        }
        guard let cut = splitIdx else {
            return Complex(real: 0, imag: parseImagMagnitude(s))
        }

        let realStr = String(s[s.startIndex..<cut])
        let imagStr = String(s[cut..<s.endIndex])
        return Complex(
            real: Double(realStr) ?? 0,
            imag: parseImagMagnitude(imagStr)
        )
    }

    /// "i" -> 1, "-i" -> -1, "2i" -> 2, "-3.5i" -> -3.5.
    private static func parseImagMagnitude(_ raw: String) -> Double {
        var s = raw
        guard let iRange = s.range(of: "i") else { return 0 }
        s.removeSubrange(iRange)
        if s.isEmpty || s == "+" { return 1 }
        if s == "-" { return -1 }
        return Double(s) ?? 0
    }
}
