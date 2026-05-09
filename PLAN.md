# FastVLM Web Pivot Plan

## Goal
Pivot the FastVLM project from a local CLI / iOS application to a modern, fully functional web application. This involves removing all legacy iOS and local testing code, wrapping the core ML inference in a FastAPI backend, and building a Next.js frontend with live webcam integration that replicates the "Active Vision" loop in the browser.

## Phase 1: Cleanup and Pruning
We will remove all files related to the old architecture to ensure a clean slate.
1. **Delete iOS/Mac App Code:**
   - `ml-fastvlm/app/` (SwiftUI, Fastlane, Xcode project).
   - `IOS_DEPLOYMENT_RUNBOOK.md`.
2. **Delete Local CLI/OpenCV Code:**
   - `ml-fastvlm/main.py` (Local OpenCV event loop).
   - `ml-fastvlm/sentinel.py` (OpenCV hardware access; logic moves to frontend).
   - `ml-fastvlm/predict.py`, `ml-fastvlm/test_vision.py` (Old headless tests).
3. **Delete Export/Dead Code:**
   - `ml-fastvlm/model_export/` (CoreML conversion is no longer needed for a web backend).
   - `models/` directory (Dead ONNX exports).
   - Unused `.zip` and `.tar` archives in the root or checkpoints directory.
   - Test images (`math_test.jpg`, etc.).

## Phase 2: Backend Development (FastAPI)
The core VLM logic (`llava` package, `inference.py`, `tools.py`) remains the "Brain". We will expose it via an API.
1. **Install Dependencies:** Add `fastapi`, `uvicorn`, and `python-multipart` to `requirements.txt` or install via pip.
2. **Create `server.py`:**
   - Initialize FastAPI with CORS middleware to allow requests from the Next.js frontend (`localhost:3000`).
   - Instantiate the `EdgeAgent` on startup.
   - Create a `POST /api/analyze` endpoint that accepts a base64 encoded image string.
   - The endpoint decodes the image, passes it to `EdgeAgent.generate_stream()`, aggregates the result, and returns a structured JSON payload containing the extracted question, the reasoning/tool used, and the final answer.

## Phase 3: Frontend Development (Next.js)
We will build a rich, aesthetic web interface that handles the perception tier (the "Eyes").
1. **Scaffold Next.js:** Create a new `web-fastvlm` directory using Next.js, React, TypeScript, and Vanilla CSS for styling.
2. **Webcam Integration:**
   - Use `navigator.mediaDevices.getUserMedia` to display a live video feed.
   - Implement the "Sentinel" logic in JavaScript: continuously capture frames to a hidden canvas and calculate basic motion and sharpness metrics.
3. **Active Vision Loop (Client-Side):**
   - Replicate the double-gate logic: wait for significant motion (indicating a new subject), then wait for stability.
   - Once stable, select the sharpest recent frame, encode it to base64, and send it to the FastAPI backend.
4. **UI Design:**
   - A polished "HUD" overlay on the video feed showing real-time stability status.
   - A sleek side panel or bottom drawer that elegantly displays the AI's findings (e.g., "Matrix Tool Used", "Answer: 42").
   - A cooldown state after an answer is received before re-arming the camera.

## Phase 4: Integration and Validation
1. Start the FastAPI backend on port 8000.
2. Start the Next.js development server on port 3000.
3. Test the end-to-end flow: hold a written math problem or question up to the web camera, ensure the frontend detects stability, fires the request, and displays the correct routed response from the backend.