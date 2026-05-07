# TASKS_TO_COMPLETE_FIRST.md

This file documents the current state of the Agentic Vision AI system and lists the high-priority tasks for the next phase of development.

## 🚀 Current State (Completed)
- **Active Vision Loop**: `main.py` uses the camera (Sentinel) to detect motion, wait for stability, and capture frames for analysis.
- **Deterministic Tool Router**: A Python-side router in `inference.py` accurately routes tasks to specialized tools instead of relying on the small LLM's intuition.
- **Toolbox Implementation**:
  - `calculator`: Scalar math solver.
  - `matrix`: NumPy-powered complex matrix multiplier.
  - `web_search`: Live news and fact search via DuckDuckGo (news fallback included).
- **Source Attribution**: Every answer clearly labels its source (LLM Memory vs. Tool).

## ⚠️ Missing Features (High Priority)
1. **Live Speech Input**: 
   - **Status**: ❌ NOT IMPLEMENTED.
   - **Task**: Integrate a Speech-to-Text (STT) engine (e.g., Faster-Whisper or PyAudio + API) to allow verbal questions.
2. **Native Video Understanding**:
   - **Status**: ⚠️ PARTIAL (Frame-based).
   - **Task**: The system currently treats video as a series of independent photos. True video understanding (temporal context across multiple frames) is missing.
3. **8-bit Optimization**:
   - **Status**: ❌ PLANNED.
   - **Task**: Optimize the LLaVA 0.5B model using 8-bit quantization to improve inference speed on CPU for interview "flex".
4. **Contextual Memory**:
   - **Status**: ❌ NOT IMPLEMENTED.
   - **Task**: Add a sliding window of conversation history so the agent remembers previous questions/answers during a session.

## 🛠️ Infrastructure Tasks
- [ ] Add `pyaudio` and `speech_recognition` dependencies.
- [ ] Refine `inference.py` to handle both visual text and transcribed speech as simultaneous inputs.
- [ ] Implement a simple UI (OpenCV or Web) to show the camera feed with real-time HUD overlays of answers.
