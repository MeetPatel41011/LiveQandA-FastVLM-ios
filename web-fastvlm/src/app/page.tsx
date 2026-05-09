"use client";

import React, { useState, useRef } from "react";
import ActiveVisionCamera from "@/components/ActiveVisionCamera";

export default function Home() {
  const [status, setStatus] = useState<"IDLE" | "SCANNING" | "THINKING" | "ANSWERING">("IDLE");
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [backendUrl, setBackendUrl] = useState("");
  const clearTimerRef = useRef<NodeJS.Timeout | null>(null);

  const handleCaptureStart = () => {
    if (clearTimerRef.current) clearTimeout(clearTimerRef.current);
    setStatus("SCANNING");
    setResult(null);
    setError(null);
  };

  const handleStableFrame = async (base64Image: string) => {
    if (!backendUrl) return; // Guard clause

    try {
      const response = await fetch(`${backendUrl}/api/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "bypass-tunnel-reminder": "true",
        },
        body: JSON.stringify({ image_base64: base64Image }),
      });

      if (!response.ok) throw new Error(`Server error: ${response.status}`);

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No response body");

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const data = JSON.parse(line.slice(6));
            
            if (data.status === "thinking") {
              setStatus("THINKING");
            } else if (data.status === "answering") {
              setStatus("ANSWERING");
              setResult(data.full_text);
            } else if (data.status === "complete") {
              setResult(data.full_text);
              // Start the 60-second persistence timer
              clearTimerRef.current = setTimeout(() => {
                setResult(null);
                setStatus("IDLE");
              }, 60000);
            } else if (data.status === "error") {
              setError(data.message);
            }
          } catch (e) {
            console.error("Parse error:", e);
          }
        }
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || "An error occurred");
      setStatus("IDLE");
    }
  };

  const handleStop = () => {
    if (clearTimerRef.current) clearTimeout(clearTimerRef.current);
    setStatus("IDLE");
    setResult(null);
    setError(null);
  };

  return (
    <main className="main-container">
      <div className="settings-overlay">
        <input 
          type="text" 
          value={backendUrl} 
          onChange={(e) => setBackendUrl(e.target.value)}
          placeholder="Backend URL (e.g. https://...loca.lt)"
          className="backend-input"
        />
      </div>
      
      <ActiveVisionCamera 
        onCaptureStart={handleCaptureStart}
        onStableFrame={handleStableFrame} 
        onStop={handleStop}
        isProcessing={status === "THINKING" || status === "ANSWERING" || status === "SCANNING"} 
        shouldEnable={backendUrl.length > 5}
      />

      {/* Setup Prompt */}
      {!backendUrl && (
        <div className="instruction-box setup-box">
          <div className="instruction-icon">🔗</div>
          <h2>Connect to Backend</h2>
          <p>Please paste your Localtunnel URL from Colab in the box at the top right to start the camera.</p>
        </div>
      )}

      {/* Instruction Box - Only shows if connected, idle and no result */}
      {backendUrl && status === "IDLE" && !result && !error && (
        <div className="instruction-box">
          <div className="instruction-icon">👁️</div>
          <h2>Waiting for Question</h2>
          <p>Hold a written or verbal question steady in front of the camera to get an answer.</p>
        </div>
      )}
      
      {/* Result Bottom Sheet */}
      <div className={`result-drawer ${(result || error || status !== "IDLE") ? "open" : ""}`}>
        <div className="drawer-handle"></div>
        <div className="status-banner">
          <span className={`status-dot ${status.toLowerCase()}`}></span>
          {status === "IDLE" ? "System Ready" : (
            <>
              {status}... 
              <span className="step-counter">
                ({status === "SCANNING" ? "1/4" : 
                  status === "THINKING" ? "2/4" : 
                  status === "ANSWERING" ? "3/4" : "4/4"})
              </span>
            </>
          )}
        </div>
        
        {error ? (
          <div className="error-message">⚠️ {error}</div>
        ) : result ? (
          <div className="result-content">
            <pre>{result}</pre>
          </div>
        ) : (
          <div className="placeholder-text">
            {status === "SCANNING" ? "Capturing pixels..." : "AI is thinking..."}
          </div>
        )}
      </div>
    </main>
  );
}
