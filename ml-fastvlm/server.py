import os
import base64
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Import the EdgeAgent from inference.py
try:
    from inference import EdgeAgent
except ImportError:
    print("Warning: Could not import EdgeAgent. Ensure you are in the ml-fastvlm directory.")
    EdgeAgent = None

app = FastAPI(title="FastVLM Agent API")

# Allow Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the Next.js domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
agent = None

@app.get("/")
async def root():
    return {"status": "ready", "agent_initialized": agent is not None}

@app.on_event("startup")
async def startup_event():
    global agent
    if EdgeAgent is not None:
        model_path = os.getenv("MODEL_PATH", "checkpoints/llava-fastvithd_0.5b_stage3")
        print(f"Initializing EdgeAgent with model: {model_path}...")
        agent = EdgeAgent(model_path=model_path)
    else:
        print("Warning: EdgeAgent not initialized.")

class AnalyzeRequest(BaseModel):
    image_base64: str # data:image/jpeg;base64,... OR raw base64
    prompt: Optional[str] = "Read the text in the image. Write the factual answer to that question. Do NOT transcribe the question itself."

class AnalyzeResponse(BaseModel):
    result: str

def decode_base64_image(base64_string: str) -> np.ndarray:
    # Remove header if present (e.g., "data:image/jpeg;base64,")
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_cv2

from fastapi.responses import StreamingResponse
import asyncio
import json

@app.post("/api/analyze")
async def analyze_image(request: AnalyzeRequest):
    if agent is None:
        raise HTTPException(status_code=500, detail="EdgeAgent is not initialized.")
    
    async def event_generator():
        try:
            # Decode image
            frame = decode_base64_image(request.image_base64)
            if frame is None:
                yield "data: " + json.dumps({"error": "Could not decode image"}) + "\n\n"
                return
            
            # 1. Immediate acknowledgement
            yield "data: " + json.dumps({"status": "thinking", "message": "Vision engine active..."}) + "\n\n"
            
            # 2. Run inference (consuming the generator)
            # We wrap the blocking generator in an async-friendly way
            stream = agent.generate_stream(image=frame, prompt=request.prompt)
            
            full_response = ""
            for chunk in stream:
                full_response += chunk
                # Send the partial response to the frontend
                yield "data: " + json.dumps({"status": "answering", "chunk": chunk, "full_text": full_response}) + "\n\n"
                await asyncio.sleep(0.01) # Yield control
                
            yield "data: " + json.dumps({"status": "complete", "full_text": full_response.strip()}) + "\n\n"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield "data: " + json.dumps({"status": "error", "message": str(e)}) + "\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
