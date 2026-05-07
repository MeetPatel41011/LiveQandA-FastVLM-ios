import cv2
import numpy as np
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2 

@dataclass
class FrameData:
    """Struct to hold a frame and its metadata."""
    image: np.ndarray
    timestamp: float
    sharpness_score: float
    motion_score: float = 0.0 # Track pixel delta from previous frame

class Sentinel:
    def __init__(self, camera_index=0, buffer_size=15, blur_threshold=100.0):
        """
        Tier 1 Controller.
        - buffer_size: 15 frames @ 30fps = 0.5 seconds of 'memory'.
        - blur_threshold: Variance score below this is considered 'blurry'.
        """
        self.cap = cv2.VideoCapture(camera_index)
        
        # Apple Engineering Note: Using a deque (double-ended queue) acts as a
        # "Ring Buffer". When we append to a full deque, old items pop off automatically.
        # This prevents memory leaks.
        self.frame_buffer = deque(maxlen=buffer_size)
        
        self.blur_threshold = blur_threshold
        self.running = False
        self.lock = threading.Lock() # Mutex for thread safety
        
        # State flags for "Barge-In" (Scenario E)
        self.is_user_speaking = False
        
        # Check camera hardware
        if not self.cap.isOpened():
            raise RuntimeError("❌ Camera Access Failed. Check permissions.")

    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """
        Cheap Computer Vision (Scenario D).
        Uses Laplacian Variance to measure edge crispness.
        Cost: <1ms on CPU.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # resize for speed - we don't need 4k for sharpness check
        small = cv2.resize(gray, (256, 256)) 
        score = cv2.Laplacian(small, cv2.CV_64F).var()
        return score

    def _camera_loop(self):
        """Background thread to act as the hardware driver."""
        prev_gray = None
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # 1. Compute metrics immediately (Tier 1 logic)
            sharpness = self._calculate_sharpness(frame)
            
            # Compute Motion (Delta from last frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (64, 64))
            motion = 0.0
            if prev_gray is not None:
                # Calculate mean absolute difference between pixels
                motion = np.mean(cv2.absdiff(gray_small, prev_gray))
            prev_gray = gray_small
            
            # 2. Store raw frame in thread-safe buffer
            with self.lock:
                self.frame_buffer.append(FrameData(
                    image=frame, # Keep this pristine!
                    timestamp=time.time(),
                    sharpness_score=sharpness,
                    motion_score=motion
                ))
            
            # Draw UX overlays ON A COPY so the user sees it, but the AI doesn't!
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Motion: {motion:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Sharpness: {sharpness:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            if motion > 8.0:
                cv2.putText(display_frame, "DETECTING NEW CONTENT...", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif self.is_camera_stable(history=8, max_motion=4.5):
                cv2.putText(display_frame, "STABLE (READY TO READ)", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Show the frame in a window named "Camera Feed"
            cv2.imshow("Camera Feed", display_frame)

    # Required: Wait 1ms for a key press so the GUI can update
    # Press 'q' to quit the loop manually
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def start(self):
        """Boots the hardware threads."""
        self.running = True
        self.cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.cam_thread.start()
        print("👁️ Sentinel Active: Monitoring Camera Buffer...")

    def stop(self):
        """Clean shutdown (Crucial for resource management)."""
        self.running = False
        if self.cam_thread.is_alive():
            self.cam_thread.join()
        self.cap.release()
        print("💤 Sentinel Asleep.")

    def get_best_frame(self) -> Optional[np.ndarray]:
        """
        The 'Magic' Function (Scenario D).
        Instead of returning the current frame, it looks back in time
        to find the sharpest image in the buffer.
        """
        with self.lock:
            if not self.frame_buffer:
                return None
            
            # Find frame with max sharpness score
            best_frame_data = max(self.frame_buffer, key=lambda x: x.sharpness_score)
            
            current_sharpness = self.frame_buffer[-1].sharpness_score
            best_sharpness = best_frame_data.sharpness_score
            
            # UX Debugging Log (Apple style: verbose logging during dev)
            # print(f"DEBUG: Current Sharpness: {current_sharpness:.1f} | Best in Buffer: {best_sharpness:.1f}")

            if best_sharpness < self.blur_threshold:
                # Scenario C: Ambiguous Visuals
                # Even the best frame is bad. Return None or raise specific warning.
                print("⚠️ Warning: Visual input is too blurry.")
                return best_frame_data.image # Return it anyway, but we logged the warning.

            return best_frame_data.image

    def is_camera_stable(self, history=5, max_motion=5.0) -> bool:
        """
        Scenario F: Active Vision.
        Check if the last `history` frames had very little motion,
        meaning the user is holding the paper steady.
        """
        with self.lock:
            if len(self.frame_buffer) < history:
                return False
            
            recent_frames = list(self.frame_buffer)[-history:]
            avg_motion = sum(f.motion_score for f in recent_frames) / history
            return avg_motion < max_motion

    def has_new_content(self, baseline_motion=15.0) -> bool:
        """Did something big happen recently (like waving a paper)?"""
        with self.lock:
            if not self.frame_buffer: return False
            # If any recent frame had massive motion, something entered the scene
            for f in reversed(self.frame_buffer):
                if f.motion_score > baseline_motion:
                    return True
            return False

# Quick test to verify hardware access
if __name__ == "__main__":
    eye = Sentinel()
    eye.start()
    try:
        # Simulate the system running for 5 seconds
        for i in range(5):
            time.sleep(1)
            best = eye.get_best_frame()
            if best is not None:
                print(f"[{i}] Retrieved optimized frame shape: {best.shape}")
    except KeyboardInterrupt:
        pass
    finally:
        eye.stop()