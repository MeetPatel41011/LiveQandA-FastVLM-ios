"use client";

import React, { useState } from "react";
import ActiveVisionCamera from "@/components/ActiveVisionCamera";

export default function Home() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [backendUrl, setBackendUrl] = useState("http://localhost:8000");

  const handleStableFrame = async (base64Image: string) => {
    setIsProcessing(true);
    setResult(null);
    setError(null);

    try {
      const response = await fetch(`${backendUrl}/api/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "bypass-tunnel-reminder": "true", // Required for localtunnel
        },
        body: JSON.stringify({ image_base64: base64Image }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data.result);
    } catch (err: any) {
      console.error(err);
      setError(err.message || "An error occurred");
    } finally {
      setIsProcessing(false);
    }
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
      <ActiveVisionCamera onStableFrame={handleStableFrame} isProcessing={isProcessing} />
      
      {/* Result Bottom Sheet */}
      <div className={`result-drawer ${result || error ? "open" : ""}`}>
        <div className="drawer-handle"></div>
        {error ? (
          <div className="error-message">⚠️ {error}</div>
        ) : result ? (
          <div className="result-content">
            <pre>{result}</pre>
          </div>
        ) : null}
      </div>
    </main>
  );
}
