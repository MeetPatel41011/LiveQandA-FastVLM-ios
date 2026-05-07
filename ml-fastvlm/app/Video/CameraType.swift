//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import Foundation

public enum CameraType: String, CaseIterable, Sendable {
    /// Iris OS: Sentinel-gated auto Q&A. Hands-free; the app fires when
    /// the camera is stable and a new question appears (with optional
    /// rolling speech transcript).
    case iris

    /// Stock continuous mode: re-runs the model on every frame.
    case continuous

    /// Manual capture: tap the shutter button to ask once.
    case single
}
