import time
import threading
import sys
from typing import Optional

# Import our custom modules
from sentinel import Sentinel

class IrisSystem:
    def __init__(self):
        print("🍏 Booting Iris OS (Active Vision Mode)...")
        print("⏳ Loading PyTorch and dependencies into memory (this will take 15-30 seconds)")
        
        # Move import down here so the print executes INSTANTLY
        from inference import EdgeAgent

        # 1. Initialize Subsystems (Universal Hardware Detection)
        self.eye = Sentinel(camera_index=0)
        self.brain = EdgeAgent() # Automatically chooses CUDA, MPS, or CPU
        
        # 2. State Management 
        self.stop_generation = threading.Event()
        self.is_generating = False

        # Cooldown prevents it from answering the same text 100 times
        self.last_inference_time = 0 
        self.cooldown_seconds = 5.0 # Wait 5 seconds after answering before looking again

    def run(self):
        """The Main Event Loop (Active Vision)"""
        self.eye.start()
        print("\n✨ System Ready. (Hold a written question up to the camera!)\n")

        try:
            while True:
                # Active Polling Loop
                time.sleep(0.5) # Check state twice a second
                
                if self.is_generating:
                    continue # Let the LLM finish talking!
                    
                # Enforce cooldown so it doesn't spam
                if time.time() - self.last_inference_time < self.cooldown_seconds:
                    continue

                # Scenario F: Active Vision Triggers
                # 1. Did something happen in front of the camera?
                if self.eye.has_new_content(baseline_motion=8.0):
                    # 2. Is the user holding the paper steady now?
                    # Increased from 3.0 to 4.5 because holding a phone causes hand shake
                    if self.eye.is_camera_stable(history=8, max_motion=4.5):
                        print("\n👁️  Motion settled. Analyzing visual context...")
                        
                        frame = self.eye.get_best_frame()
                        if frame is not None:
                            self.is_generating = True
                            
                            # Instruction Tuning: We must give it an explicit command.
                            # "Please answer" implies a conversational chat. We want a machine output.
                            strict_query = "Read the text in the image. Write the factual answer to that question. Do NOT transcribe the question itself."
                            self._run_inference(frame, strict_query)
                            
                            self.last_inference_time = time.time()
                            self.is_generating = False
                            print("\n👀 Resuming observation...")

        except KeyboardInterrupt:
            print("\n🍎 Shutting down...")
            self.eye.stop()
            sys.exit(0)

    def _run_inference(self, frame, query):
        """Helper to run the stream and print to terminal."""
        # The AI now yields Agentic tool responses instead of token streams
        stream = self.brain.generate_stream(
            image=frame, 
            prompt=query, 
            stop_event=self.stop_generation
        )
        
        for generated_block in stream:
            print(generated_block, end="", flush=True)
        print("\n")

if __name__ == "__main__":
    app = IrisSystem()
    app.run()