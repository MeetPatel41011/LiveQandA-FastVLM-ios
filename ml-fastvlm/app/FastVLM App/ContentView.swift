//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import AVFoundation
import MLXLMCommon
import SwiftUI
import Video

// support swift 6
extension CVImageBuffer: @unchecked @retroactive Sendable {}
extension CMSampleBuffer: @unchecked @retroactive Sendable {}

// delay between frames -- controls the frame rate of the updates
let FRAME_DELAY = Duration.milliseconds(1)

struct ContentView: View {
    @State private var camera = CameraController()
    @State private var model: FastVLMModel
    @State private var agent: IrisAgent
    @State private var sentinel = Sentinel()
    @State private var speech = SpeechController()

    init() {
        // Construct model + agent eagerly so the agent always sees the
        // same FastVLMModel instance the rest of the view interacts with.
        let m = FastVLMModel()
        _model = State(initialValue: m)
        _agent = State(initialValue: IrisAgent(model: m))
    }

    /// stream of frames -> VideoFrameView, see distributeVideoFrames
    @State private var framesToDisplay: AsyncStream<CVImageBuffer>?

    @State private var prompt = "Describe the image in English."
    @State private var promptSuffix = "Output should be brief, about 15 words or less."

    @State private var isShowingInfo: Bool = false

    /// Default to Iris (auto Sentinel-gated Q&A). Users can fall back to
    /// the stock continuous / single modes via the picker.
    @State private var selectedCameraType: CameraType = .iris
    @State private var isEditingPrompt: Bool = false

    /// PerfHUD visibility — toggled by a long-press on the camera view.
    @State private var showPerfHUD: Bool = false

    /// Whether speech permission has been requested this session.
    @State private var speechAuthRequested: Bool = false

    var toolbarItemPlacement: ToolbarItemPlacement {
        var placement: ToolbarItemPlacement = .navigation
        #if os(iOS)
        placement = .topBarLeading
        #endif
        return placement
    }
    
    var body: some View {
        NavigationStack {
            Form {
                Section {
                    VStack(alignment: .leading, spacing: 10.0) {
                        Picker("Camera Type", selection: $selectedCameraType) {
                            ForEach(CameraType.allCases, id: \.self) { cameraType in
                                Text(cameraType.rawValue.capitalized).tag(cameraType)
                            }
                        }
                        // Prevent macOS from adding a text label for the picker
                        .labelsHidden()
                        .pickerStyle(.segmented)
                        .onChange(of: selectedCameraType) { _, _ in
                            // Cancel any in-flight requests + reset Iris
                            // state when switching modes.
                            model.cancel()
                            agent.reset()
                            speech.stop()
                        }

                        if let framesToDisplay {
                            VideoFrameView(
                                frames: framesToDisplay,
                                cameraType: selectedCameraType,
                                action: { frame in
                                    processSingleFrame(frame)
                                })
                                // Because we're using the AVCaptureSession preset
                                // `.vga640x480`, we can assume this aspect ratio
                                .aspectRatio(4/3, contentMode: .fit)
                                #if os(macOS)
                                .frame(maxWidth: 750)
                                #endif
                                .overlay(alignment: .top) {
                                    if !model.promptTime.isEmpty {
                                        Text("TTFT \(model.promptTime)")
                                            .font(.caption)
                                            .foregroundStyle(.white)
                                            .monospaced()
                                            .padding(.vertical, 4.0)
                                            .padding(.horizontal, 6.0)
                                            .background(alignment: .center) {
                                                RoundedRectangle(cornerRadius: 8)
                                                    .fill(Color.black.opacity(0.6))
                                            }
                                            .padding(.top)
                                    }
                                }
                                .overlay(alignment: .topLeading) {
                                    if showPerfHUD {
                                        PerfHUD(
                                            ttftMs: agent.lastTTFTms,
                                            sharpness: sentinel.lastSharpness,
                                            motion: sentinel.lastMotion,
                                            isStable: sentinel.isStable(),
                                            hasNewContent: sentinel.hasNewContent(),
                                            cooldownRemaining: agent.cooldown.remaining
                                        )
                                        .padding(8)
                                        .transition(.opacity)
                                    }
                                }
                                .onLongPressGesture(minimumDuration: 0.8) {
                                    withAnimation { showPerfHUD.toggle() }
                                }
                                #if !os(macOS)
                                .overlay(alignment: .topTrailing) {
                                    CameraControlsView(
                                        backCamera: $camera.backCamera,
                                        device: $camera.device,
                                        devices: $camera.devices)
                                    .padding()
                                }
                                #endif
                                .overlay(alignment: .bottom) {
                                    if selectedCameraType == .continuous || selectedCameraType == .iris {
                                        statusCapsule
                                            .padding(.bottom)
                                    }
                                }
                                #if os(macOS)
                                .frame(maxWidth: .infinity)
                                .frame(minWidth: 500)
                                .frame(minHeight: 375)
                                #endif
                        }
                    }
                }
                .listRowInsets(EdgeInsets())
                .listRowBackground(Color.clear)
                .listRowSeparator(.hidden)

                promptSections

                responseSection

                #if os(macOS)
                Spacer()
                #endif
            }
            
            #if os(iOS)
            .listSectionSpacing(0)
            #elseif os(macOS)
            .padding()
            #endif
            .task {
                camera.start()
            }
            .task {
                await model.load()
            }
            .task(id: speechAuthRequested) {
                if !speechAuthRequested {
                    await speech.requestAuthorization()
                    speechAuthRequested = true
                }
            }

            #if !os(macOS)
            .onAppear {
                // Prevent the screen from dimming or sleeping due to inactivity
                UIApplication.shared.isIdleTimerDisabled = true
            }
            .onDisappear {
                // Resumes normal idle timer behavior
                UIApplication.shared.isIdleTimerDisabled = false
            }
            #endif

            // task to distribute video frames -- this will cancel
            // and restart when the view is on/off screen.  note: it is
            // important that this is here (attached to the VideoFrameView)
            // rather than the outer view because this has the correct lifecycle
            .task {
                if Task.isCancelled {
                    return
                }

                await distributeVideoFrames()
            }

            .navigationTitle("FastVLM")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: toolbarItemPlacement) {
                    Button {
                        isShowingInfo.toggle()
                    }
                    label: {
                        Image(systemName: "info.circle")
                    }
                }

                ToolbarItem(placement: .primaryAction) {
                    if isEditingPrompt {
                        Button {
                            isEditingPrompt.toggle()
                        }
                        label: {
                            Text("Done")
                                .fontWeight(.bold)
                        }
                    }
                    else {
                        Menu {
                            Button("Describe image") {
                                prompt = "Describe the image in English."
                                promptSuffix = "Output should be brief, about 15 words or less."
                            }
                            Button("Facial expression") {
                                prompt = "What is this person's facial expression?"
                                promptSuffix = "Output only one or two words."
                            }
                            Button("Read text") {
                                prompt = "What is written in this image?"
                                promptSuffix = "Output only the text in the image."
                            }
                            #if !os(macOS)
                            Button("Customize...") {
                                isEditingPrompt.toggle()
                            }
                            #endif
                        } label: { Text("Prompts") }
                    }
                }

                #if os(iOS)
                ToolbarItem(placement: .primaryAction) {
                    micButton
                }
                #endif
            }
            .sheet(isPresented: $isShowingInfo) {
                InfoView()
            }
        }
    }

    var promptSummary: some View {
        Section("Prompt") {
            VStack(alignment: .leading, spacing: 4.0) {
                let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmedPrompt.isEmpty {
                    Text(trimmedPrompt)
                        .foregroundStyle(.secondary)
                }

                let trimmedSuffix = promptSuffix.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmedSuffix.isEmpty {
                    Text(trimmedSuffix)
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
            }
        }
    }

    var promptForm: some View {
        Group {
            #if os(iOS)
            Section("Prompt") {
                TextEditor(text: $prompt)
                    .frame(minHeight: 38)
            }

            Section("Prompt Suffix") {
                TextEditor(text: $promptSuffix)
                    .frame(minHeight: 38)
            }
            #elseif os(macOS)
            Section {
                HStack(alignment: .top) {
                    VStack(alignment: .leading) {
                        Text("Prompt")
                            .font(.headline)

                        TextEditor(text: $prompt)
                            .frame(height: 38)
                            .padding(.horizontal, 8.0)
                            .padding(.vertical, 10.0)
                            .background(Color(.textBackgroundColor))
                            .cornerRadius(10.0)
                    }

                    VStack(alignment: .leading) {
                        Text("Prompt Suffix")
                            .font(.headline)

                        TextEditor(text: $promptSuffix)
                            .frame(height: 38)
                            .padding(.horizontal, 8.0)
                            .padding(.vertical, 10.0)
                            .background(Color(.textBackgroundColor))
                            .cornerRadius(10.0)
                    }
                }
            }
            .padding(.vertical)
            #endif
        }
    }

    var promptSections: some View {
        Group {
            #if os(iOS)
            if isEditingPrompt {
                promptForm
            }
            else {
                promptSummary
            }
            #elseif os(macOS)
            promptForm
            #endif
        }
    }

    // MARK: - Iris helper views

    /// Bottom status pill on the camera view. Dispatches on mode + state
    /// so .iris and .continuous can share visual language.
    @ViewBuilder
    private var statusCapsule: some View {
        let (label, system, fg, bg): (String, String?, Color, Color) = {
            if selectedCameraType == .iris {
                if agent.isThinking {
                    return ("Thinking…", nil, .black, .yellow)
                }
                if agent.cooldown.isActive {
                    return ("Answer ready", "lightbulb.fill", .white, .green)
                }
                if sentinel.isStable() && sentinel.hasNewContent() {
                    return ("Reading…", nil, .black, .yellow)
                }
                return ("Watching", "eye.fill", .white, .gray)
            } else {
                switch model.evaluationState {
                case .idle:
                    return (model.evaluationState.rawValue, "clock.fill", .white, .gray)
                case .processingPrompt:
                    return (model.evaluationState.rawValue, nil, .black, .yellow)
                case .generatingResponse:
                    return (model.evaluationState.rawValue, "lightbulb.fill", .white, .green)
                }
            }
        }()

        HStack(spacing: 6) {
            if let system {
                Image(systemName: system).font(.caption)
            } else if selectedCameraType != .iris || agent.isThinking {
                ProgressView().tint(fg).controlSize(.small)
            }
            Text(label)
        }
        .foregroundStyle(fg)
        .font(.caption)
        .bold()
        .padding(.vertical, 6)
        .padding(.horizontal, 8)
        .background(bg)
        .clipShape(.capsule)
    }

    /// Response panel. Iris modes show the routed answer + Source badge;
    /// .continuous keeps the raw streaming output for backwards compat.
    @ViewBuilder
    private var responseSection: some View {
        Section {
            if selectedCameraType == .continuous {
                if model.output.isEmpty && model.running {
                    ProgressView()
                        .controlSize(.large)
                        .frame(maxWidth: .infinity)
                } else {
                    ScrollView {
                        Text(model.output)
                            .foregroundStyle(isEditingPrompt ? .secondary : .primary)
                            .textSelection(.enabled)
                            #if os(macOS)
                            .font(.headline)
                            .fontWeight(.regular)
                            #endif
                    }
                    .frame(minHeight: 50.0, maxHeight: 200.0)
                }
            } else {
                irisResponseBody
            }
        } header: {
            Text("Response")
                #if os(macOS)
                .font(.headline)
                .padding(.bottom, 2.0)
                #endif
        }
    }

    @ViewBuilder
    private var irisResponseBody: some View {
        if agent.isThinking && agent.lastAnswer == nil {
            ProgressView()
                .controlSize(.large)
                .frame(maxWidth: .infinity)
        } else if let routed = agent.lastAnswer {
            VStack(alignment: .leading, spacing: 8) {
                ScrollView {
                    Text(routed.body)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(minHeight: 50.0, maxHeight: 200.0)

                Text("Source: \(routed.source)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        } else if let err = agent.errorMessage {
            Text(err).foregroundStyle(.red)
        } else {
            Text("Hold a written question up to the camera, or tap the mic to ask aloud.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(minHeight: 50.0, maxHeight: .infinity)
        }
    }

    /// Microphone button. Press-and-hold to listen; release to commit
    /// transcript. While Iris mode is on the user can still hold the
    /// mic — its transcript is appended to the next inference fire.
    #if os(iOS)
    @ViewBuilder
    private var micButton: some View {
        Image(systemName: speech.isListening
              ? "mic.fill"
              : "mic")
            .font(.title3)
            .foregroundStyle(speech.isListening ? .red : .primary)
            .padding(8)
            .contentShape(Rectangle())
            .onLongPressGesture(
                minimumDuration: 0.15,
                pressing: { isPressing in
                    if isPressing {
                        if speech.authorizationStatus == .authorized {
                            speech.clearTranscript()
                            speech.start()
                        }
                    } else {
                        speech.stop()
                        // In single mode, releasing the mic captures the
                        // *current* frame as the visual context.
                        if selectedCameraType == .single,
                           let last = sentinel.bestFrame() {
                            processSingleFrame(last)
                        }
                    }
                },
                perform: {}
            )
            .accessibilityLabel(speech.isListening ? "Listening" : "Hold to ask")
    }
    #endif

    /// Legacy free-form-chat analyser used by .continuous mode. Every
    /// frame triggers a new inference; output streams as raw text.
    @MainActor
    func analyzeVideoFrames(_ frames: AsyncStream<CVImageBuffer>) async {
        for await frame in frames {
            let userInput = UserInput(
                prompt: .text("\(prompt) \(promptSuffix)"),
                images: [.ciImage(CIImage(cvPixelBuffer: frame))]
            )

            let t = await model.generate(userInput)
            _ = await t.result

            do {
                try await Task.sleep(for: FRAME_DELAY)
            } catch { return }
        }
    }

    /// Iris mode analyser: Sentinel-gated, routed through IrisAgent.
    /// Fires only when the camera is stable AND something new entered
    /// the scene AND the cooldown has elapsed.
    @MainActor
    func analyzeIrisFrames(_ frames: AsyncStream<CVImageBuffer>) async {
        for await frame in frames {
            // Always update Sentinel so the gates + PerfHUD have data.
            sentinel.update(frame)

            guard !agent.isThinking else { continue }
            guard !agent.cooldown.isActive else { continue }
            guard sentinel.hasNewContent() && sentinel.isStable() else { continue }

            let best = sentinel.bestFrame() ?? frame
            let transcript = speech.transcript.isEmpty ? nil : speech.transcript

            await agent.ask(frame: best, spokenTranscript: transcript)

            // Avoid immediately re-firing on the same paper. The next
            // hasNewContent() call will only return true once motion
            // exceeds the baseline again.
            sentinel.clearMotionHistory()

            // Yield briefly so the response renders.
            try? await Task.sleep(for: FRAME_DELAY)
        }
    }

    func distributeVideoFrames() async {
        let frames = AsyncStream<CMSampleBuffer>(bufferingPolicy: .bufferingNewest(1)) {
            camera.attach(continuation: $0)
        }

        let (framesToDisplay, framesToDisplayContinuation) = AsyncStream.makeStream(
            of: CVImageBuffer.self,
            bufferingPolicy: .bufferingNewest(1)
        )
        self.framesToDisplay = framesToDisplay

        let (framesToAnalyze, framesToAnalyzeContinuation) = AsyncStream.makeStream(
            of: CVImageBuffer.self,
            bufferingPolicy: .bufferingNewest(1)
        )

        async let distributeFrames: () = {
            for await sampleBuffer in frames {
                if let frame = sampleBuffer.imageBuffer {
                    framesToDisplayContinuation.yield(frame)
                    // Both .continuous and .iris consume the analyse
                    // stream; .single only fires on the shutter button.
                    let mode = await selectedCameraType
                    if mode == .continuous || mode == .iris {
                        framesToAnalyzeContinuation.yield(frame)
                    }
                }
            }

            await MainActor.run {
                self.framesToDisplay = nil
                self.camera.detatch()
            }

            framesToDisplayContinuation.finish()
            framesToAnalyzeContinuation.finish()
        }()

        switch selectedCameraType {
        case .iris:
            async let analyze: () = analyzeIrisFrames(framesToAnalyze)
            await distributeFrames
            await analyze
        case .continuous:
            async let analyze: () = analyzeVideoFrames(framesToAnalyze)
            await distributeFrames
            await analyze
        case .single:
            await distributeFrames
        }
    }

    /// Manual capture (single mode). Routes through IrisAgent so the
    /// user gets the same JSON-extracted answer + Source badge as in
    /// Iris mode, but only on explicit tap.
    func processSingleFrame(_ frame: CVImageBuffer) {
        Task { @MainActor in
            agent.reset()
            let transcript = speech.transcript.isEmpty ? nil : speech.transcript
            await agent.ask(frame: frame, spokenTranscript: transcript)
        }
    }
}

#Preview {
    ContentView()
}
