//
// SpeechController.swift
// On-device speech-to-text using Apple's Speech framework.
//
// Two operating modes:
//
//   - Push-to-talk: caller invokes start()/stop() via a hold gesture on
//     the mic button. transcript is the final text after stop().
//   - Continuous (Auto/Iris mode): caller invokes start() once when the
//     view appears; transcript is the rolling latest hypothesis at the
//     instant Sentinel decides to fire inference.
//
// requiresOnDeviceRecognition = true means:
//   - Free (no Apple-side network calls or quotas).
//   - Private (audio never leaves the device).
//   - Slightly less accurate than the cloud path on long utterances —
//     fine for short questions ("what is the capital of Japan").
//
// macOS gets a no-op implementation so the existing macOS build path
// keeps working.
//

import Foundation

#if os(iOS)

import AVFoundation
import Speech

@Observable
@MainActor
final class SpeechController {

    // MARK: - Public state

    /// The latest finalised + intermediate transcript.
    var transcript: String = ""

    /// True while the audio engine is feeding the recogniser.
    var isListening: Bool = false

    /// User-visible error string (nil if all is well).
    var errorMessage: String?

    /// Authorisation state at the moment of the last `requestAuthorization`.
    var authorizationStatus: SFSpeechRecognizerAuthorizationStatus = .notDetermined

    // MARK: - Private state

    private let recognizer: SFSpeechRecognizer? = SFSpeechRecognizer(locale: .current)
    private let audioEngine = AVAudioEngine()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?

    // MARK: - Authorisation

    func requestAuthorization() async {
        let speechStatus: SFSpeechRecognizerAuthorizationStatus = await withCheckedContinuation { cont in
            SFSpeechRecognizer.requestAuthorization { status in cont.resume(returning: status) }
        }
        authorizationStatus = speechStatus

        // Microphone access is also required for AVAudioEngine.
        if AVAudioApplication.shared.recordPermission == .undetermined {
            _ = await AVAudioApplication.requestRecordPermission()
        }
    }

    // MARK: - Lifecycle

    func start() {
        guard !isListening else { return }
        guard let recognizer, recognizer.isAvailable else {
            errorMessage = "Speech recognition is not available on this device."
            return
        }
        guard authorizationStatus == .authorized else {
            errorMessage = "Speech permission not granted."
            return
        }

        do {
            try configureSession()
            try beginRecognition(with: recognizer)
            isListening = true
            errorMessage = nil
        } catch {
            errorMessage = "Failed to start listening: \(error.localizedDescription)"
            stop()
        }
    }

    func stop() {
        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
        }
        request?.endAudio()
        request = nil
        task?.cancel()
        task = nil

        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        isListening = false
    }

    /// Reset the rolling transcript without stopping listening.
    func clearTranscript() {
        transcript = ""
    }

    // MARK: - Internals

    private func configureSession() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record,
                                mode: .measurement,
                                options: [.duckOthers])
        try session.setActive(true, options: .notifyOthersOnDeactivation)
    }

    private func beginRecognition(with recognizer: SFSpeechRecognizer) throws {
        let req = SFSpeechAudioBufferRecognitionRequest()
        req.shouldReportPartialResults = true
        if recognizer.supportsOnDeviceRecognition {
            req.requiresOnDeviceRecognition = true
        }
        request = req

        let inputNode = audioEngine.inputNode
        let format = inputNode.outputFormat(forBus: 0)

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            self?.request?.append(buffer)
        }

        audioEngine.prepare()
        try audioEngine.start()

        task = recognizer.recognitionTask(with: req) { [weak self] result, error in
            guard let self else { return }
            if let result {
                Task { @MainActor in
                    self.transcript = result.bestTranscription.formattedString
                }
            }
            if let error {
                Task { @MainActor in
                    // Cancellation of the task throws here too — only
                    // surface real errors to the UI.
                    if (error as NSError).code != 203 {
                        self.errorMessage = error.localizedDescription
                    }
                }
            }
            if result?.isFinal == true || error != nil {
                Task { @MainActor in self.stop() }
            }
        }
    }
}

#else

// macOS / preview stub. Compiles without dragging in Speech.framework.

import Foundation

@Observable
@MainActor
final class SpeechController {
    var transcript: String = ""
    var isListening: Bool = false
    var errorMessage: String? = "Speech is iOS-only in Iris."

    enum AuthorizationStatus: Int { case notDetermined, denied, restricted, authorized }
    var authorizationStatus: AuthorizationStatus = .notDetermined

    func requestAuthorization() async {}
    func start() {}
    func stop() {}
    func clearTranscript() { transcript = "" }
}

#endif
