"use client";

import React, { useState, useRef, useEffect } from "react";
import ActiveVisionCamera from "@/components/ActiveVisionCamera";

export default function Home() {
  const [status, setStatus] = useState<"IDLE" | "LOADING" | "SCANNING" | "THINKING" | "ANSWERING" | "COMPLETE">("IDLE");
  const [result, setResult] = useState<string | null>(null);
  const [perfData, setPerfData] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const clearTimerRef = useRef<NodeJS.Timeout | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const [isModelReady, setIsModelReady] = useState(false);

  // Initialize WebGPU Worker on Mount
  useEffect(() => {
    if (typeof Worker !== 'undefined') {
      setStatus("LOADING");
      workerRef.current = new Worker('/worker/inference.worker.js');
      
      workerRef.current.onmessage = async (event) => {
        const { type, success, result: outputResult, error: workerError, payload } = event.data;

        if (type === 'modelLoaded') {
          if (success) {
            setIsModelReady(true);
            setStatus("IDLE");
            console.log('Main Thread: WebGPU Model loaded successfully.');
          } else {
            setError(`Failed to load model: ${workerError}`);
            setStatus("IDLE");
          }
        } 
        else if (type === 'inferenceResult') {
          if (success) {
            setStatus("COMPLETE");
            setResult(outputResult);
            // Auto-clear after 60s
            clearTimerRef.current = setTimeout(() => {
              setResult(null);
              setStatus("IDLE");
            }, 60000);
          } else {
            setError(`Inference failed: ${workerError}`);
            setStatus("IDLE");
          }
        } 
        else if (type === 'toolCall' && payload?.type === 'web_search') {
          console.log(`Main Thread: Received toolCall from worker for query: "${payload.query}"`);
          setStatus("ANSWERING");
          setResult(`Searching live data for: "${payload.query}"...`);
          
          const startPerf = performance.now();
          try {
            // Call secure serverless proxy
            const response = await fetch(`/api/tavily?query=${encodeURIComponent(payload.query)}`);
            const data = await response.json();
            
            if (data.success) {
              workerRef.current?.postMessage({ type: 'toolResult', payload: { type: 'web_search', results: data.results } });
            } else {
              workerRef.current?.postMessage({ type: 'toolResult', payload: { type: 'web_search', error: data.error } });
            }
            
            const toolLatency = ((performance.now() - startPerf) / 1000).toFixed(3);
            setPerfData(`Web Search Latency: ${toolLatency}s`);
            
          } catch (apiError: any) {
            workerRef.current?.postMessage({ type: 'toolResult', payload: { type: 'web_search', error: apiError.message } });
          }
        }
      };

      // Instruct worker to download and cache the ONNX model into WebGPU
      workerRef.current.postMessage({ 
        type: 'loadModel', 
        payload: { task: 'image-text-to-text', modelName: 'Xenova/llava-onevision' } 
      });

      return () => {
        if (workerRef.current) workerRef.current.terminate();
      };
    } else {
      setError('Web Workers are not supported in this browser.');
    }
  }, []);

  const handleCaptureStart = () => {
    if (clearTimerRef.current) clearTimeout(clearTimerRef.current);
    setStatus("SCANNING");
    setResult(null);
    setPerfData(null);
    setError(null);
  };

  const handleStableFrame = async (base64Image: string) => {
    if (!isModelReady || !workerRef.current) return;
    setStatus("THINKING");
    
    // Send image to WebGPU Worker
    workerRef.current.postMessage({ 
      type: 'runInference', 
      payload: { imageBlob: base64Image } 
    });
  };

  const handleStop = () => {
    if (clearTimerRef.current) clearTimeout(clearTimerRef.current);
    setStatus("IDLE");
    setResult(null);
    setPerfData(null);
    setError(null);
  };

  return (
    <main className="main-container">
      {status === "LOADING" && (
        <div className="settings-overlay" style={{ background: 'rgba(0,0,0,0.8)', color: 'white', padding: '10px' }}>
          Downloading Model (300MB) to WebGPU Cache...
        </div>
      )}
      
      <ActiveVisionCamera
      onCaptureStart={handleCaptureStart}
      onStableFrame={handleStableFrame} 
      onStop={handleStop}
      isProcessing={status === "THINKING" || status === "ANSWERING" || status === "SCANNING" || status === "COMPLETE" || status === "LOADING"} 
      shouldEnable={isModelReady}
      />

      {/* Setup Prompt */}
      {!isModelReady && status !== "LOADING" && (
      <div className="instruction-box setup-box">
        <div className="instruction-icon">⚙️</div>
        <h2>Initializing WebGPU</h2>
        <p>Your browser is allocating local GPU memory for the Vision Model.</p>
      </div>
      )}

      {/* Instruction Box - Only shows if connected, idle and no result */}
      {isModelReady && status === "IDLE" && !result && !error && (
      <div className="instruction-box">
        <div className="instruction-icon">👁️</div>
        <h2>Web-Native AI Ready</h2>
        <p>Hold a question steady. Inference runs 100% locally on your GPU.</p>
      </div>
      )}

      {/* Result Bottom Sheet */}
      <div className={`result-drawer ${(result || error || status !== "IDLE" && status !== "LOADING") ? "open" : ""}`}>
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

          {perfData && (
            <div className="perf-dashboard">
              <div className="perf-header">🏎️ ML Performance Report</div>
              <div className="perf-grid">
                {perfData.split('\n').map((line, i) => {
                  const [label, val] = line.split(':');
                  return (
                    <div key={i} className="perf-stat">
                      <span className="perf-label">{label}</span>
                      <span className="perf-value">{val}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      ) : (        <div className="placeholder-text">
          {status === "SCANNING" ? "Position your question in frame..." : "AI is thinking..."}
        </div>
      )}
      </div>    </main>
  );
}
