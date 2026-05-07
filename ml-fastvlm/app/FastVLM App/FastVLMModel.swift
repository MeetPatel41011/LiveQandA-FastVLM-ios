//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import CoreImage
import FastVLM
import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import MLXVLM

@Observable
@MainActor
class FastVLMModel {

    public var running = false
    public var modelInfo = ""
    public var output = ""
    public var promptTime: String = ""

    enum LoadState {
        case idle
        case loaded(ModelContainer)
    }

    private let modelConfiguration = FastVLM.modelConfiguration

    /// parameters controlling the output
    let generateParameters = GenerateParameters(temperature: IrisConfig.temperature)

    /// Token budget. Iris asks for short JSON; legacy modes might want
    /// longer prose. Default lives in IrisConfig.maxTokens (100).
    let maxTokens = IrisConfig.maxTokens

    /// update the display every N tokens -- 4 looks like it updates continuously
    /// and is low overhead.  observed ~15% reduction in tokens/s when updating
    /// on every token
    let displayEveryNTokens = 4

    /// Time-to-first-token of the most recent inference, in milliseconds.
    /// Surface in PerfHUD; the existing `promptTime` string keeps the
    /// human-readable form for the existing UI.
    public var lastTTFTms: Int = 0

    private var loadState = LoadState.idle
    private var currentTask: Task<Void, Never>?

    enum EvaluationState: String, CaseIterable {
        case idle = "Idle"
        case processingPrompt = "Processing Prompt"
        case generatingResponse = "Generating Response"
    }

    public var evaluationState = EvaluationState.idle

    public init() {
        FastVLM.register(modelFactory: VLMModelFactory.shared)
    }

    private func _load() async throws -> ModelContainer {
        switch loadState {
        case .idle:
            // limit the buffer cache
            MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)

            let modelContainer = try await VLMModelFactory.shared.loadContainer(
                configuration: modelConfiguration
            ) {
                [modelConfiguration] progress in
                Task { @MainActor in
                    self.modelInfo =
                        "Downloading \(modelConfiguration.name): \(Int(progress.fractionCompleted * 100))%"
                }
            }
            self.modelInfo = "Loaded"
            loadState = .loaded(modelContainer)
            return modelContainer

        case .loaded(let modelContainer):
            return modelContainer
        }
    }

    public func load() async {
        do {
            _ = try await _load()
            // Cold-start warmup: run one tiny inference with a 64x64
            // black image. This pre-compiles the MLX kernels and primes
            // the GPU command queue, shaving 200-400 ms off the first
            // user-visible inference. Invisible to the UI.
            await warmup()
        } catch {
            self.modelInfo = "Error loading model: \(error)"
        }
    }

    /// One-shot dummy inference to pay cold-start cost up front.
    private func warmup() async {
        let size = 64
        let bytesPerRow = size * 4
        var pixels = [UInt8](repeating: 0, count: bytesPerRow * size)
        // Write straight black; CGContext below treats this as ARGB.
        let ci = pixels.withUnsafeMutableBytes { raw -> CIImage? in
            let cs = CGColorSpaceCreateDeviceRGB()
            guard let cg = CGContext(
                data: raw.baseAddress,
                width: size,
                height: size,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: cs,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ),
            let img = cg.makeImage()
            else { return nil }
            return CIImage(cgImage: img)
        }
        guard let dummy = ci else { return }

        let userInput = UserInput(
            prompt: .text(" "),
            images: [.ciImage(dummy)]
        )
        do {
            let modelContainer = try await _load()
            _ = try await modelContainer.perform { context in
                let input = try await context.processor.prepare(input: userInput)
                _ = try MLXLMCommon.generate(
                    input: input,
                    parameters: GenerateParameters(temperature: 0.0),
                    context: context
                ) { tokens in
                    // Stop after a single token; we only care about the
                    // forward pass timing, not the output.
                    return tokens.count >= 1 ? .stop : .more
                }
            }
        } catch {
            // Warmup failures are non-fatal.
        }
    }

    public func generate(_ userInput: UserInput) async -> Task<Void, Never> {
        if let currentTask, running {
            return currentTask
        }

        running = true
        
        // Cancel any existing task
        currentTask?.cancel()

        // Create new task and store reference
        let task = Task {
            do {
                let modelContainer = try await _load()

                // each time you generate you will get something new
                MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))
                
                // Check if task was cancelled
                if Task.isCancelled { return }

                let result = try await modelContainer.perform { context in
                    // Measure the time it takes to prepare the input
                    
                    Task { @MainActor in
                        evaluationState = .processingPrompt
                    }

                    let llmStart = Date()
                    let input = try await context.processor.prepare(input: userInput)
                    
                    var seenFirstToken = false

                    // FastVLM generates the output
                    let result = try MLXLMCommon.generate(
                        input: input, parameters: generateParameters, context: context
                    ) { tokens in
                        // Check if task was cancelled
                        if Task.isCancelled {
                            return .stop
                        }

                        if !seenFirstToken {
                            seenFirstToken = true
                            
                            // produced first token, update the time to first token,
                            // the processing state and start displaying the text
                            let llmDuration = Date().timeIntervalSince(llmStart)
                            let text = context.tokenizer.decode(tokens: tokens)
                            Task { @MainActor in
                                evaluationState = .generatingResponse
                                self.output = text
                                self.promptTime = "\(Int(llmDuration * 1000)) ms"
                            }
                        }

                        // Show the text in the view as it generates
                        if tokens.count % displayEveryNTokens == 0 {
                            let text = context.tokenizer.decode(tokens: tokens)
                            Task { @MainActor in
                                self.output = text
                            }
                        }

                        if tokens.count >= maxTokens {
                            return .stop
                        } else {
                            return .more
                        }
                    }
                    
                    // Return the duration of the LLM and the result
                    return result
                }
                
                // Check if task was cancelled before updating UI
                if !Task.isCancelled {
                    self.output = result.output
                }

            } catch {
                if !Task.isCancelled {
                    output = "Failed: \(error)"
                }
            }

            if evaluationState == .generatingResponse {
                evaluationState = .idle
            }

            running = false
        }
        
        currentTask = task
        return task
    }
    
    public func cancel() {
        currentTask?.cancel()
        currentTask = nil
        running = false
        output = ""
        promptTime = ""
    }

    // MARK: - Iris OS path

    /// Run inference and return the complete generated string. Used by
    /// the Iris pipeline (Sentinel -> PromptBuilder -> here -> JSON parse
    /// -> ToolRouter). Updates TTFT side-channel state; does NOT touch
    /// `self.output` (the Iris UI renders the *routed* answer, not the
    /// raw stream).
    public func generateForIris(_ userInput: UserInput) async throws -> String {
        let modelContainer = try await _load()
        MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))

        running = true
        evaluationState = .processingPrompt
        defer {
            running = false
            evaluationState = .idle
        }

        let llmStart = Date()
        var seenFirstToken = false
        var collected = ""

        let result = try await modelContainer.perform { [maxTokens] context in
            let input = try await context.processor.prepare(input: userInput)
            return try MLXLMCommon.generate(
                input: input,
                parameters: GenerateParameters(temperature: IrisConfig.temperature),
                context: context
            ) { tokens in
                if Task.isCancelled { return .stop }

                if !seenFirstToken {
                    seenFirstToken = true
                    let llmDuration = Date().timeIntervalSince(llmStart)
                    let ms = Int(llmDuration * 1000)
                    Task { @MainActor in
                        self.lastTTFTms = ms
                        self.promptTime = "\(ms) ms"
                        self.evaluationState = .generatingResponse
                    }
                }

                collected = context.tokenizer.decode(tokens: tokens)
                return tokens.count >= maxTokens ? .stop : .more
            }
        }

        // Prefer the final detokenisation from MLX over our incremental
        // scratch — it handles trailing partial tokens correctly.
        return result.output.isEmpty ? collected : result.output
    }
}
