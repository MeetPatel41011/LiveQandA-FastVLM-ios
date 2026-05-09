"use client";

import React, { useEffect, useRef, useState } from "react";

interface ActiveVisionCameraProps {
  onStableFrame: (base64Image: string) => void;
  onStop?: () => void;
  isProcessing: boolean;
  shouldEnable: boolean;
}

export default function ActiveVisionCamera({ onStableFrame, onStop, isProcessing, shouldEnable }: ActiveVisionCameraProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [motionScore, setMotionScore] = useState<number>(0);
  const [status, setStatus] = useState<"IDLE" | "MOTION_DETECTED" | "STABLE">("IDLE");
  const [hasMounted, setHasMounted] = useState(false);

  // Internal refs
  const stateRef = useRef({
    prevGrayData: null as Uint8ClampedArray | null,
    recentMotions: [] as number[],
    lastFrame: null as string | null,
  });

  const triggerManualCapture = async () => {
    if (!videoRef.current || isProcessing) return;
    
    const video = videoRef.current;
    const frames: { data: string; score: number }[] = [];
    
    // Step 1: Burst Capture (5 frames over 500ms)
    for (let i = 0; i < 5; i++) {
      const fullCanvas = document.createElement("canvas");
      fullCanvas.width = video.videoWidth;
      fullCanvas.height = video.videoHeight;
      const fullCtx = fullCanvas.getContext("2d");
      if (fullCtx) {
        fullCtx.drawImage(video, 0, 0);
        
        // Basic Sharpness Score
        const smallCanvas = document.createElement("canvas");
        smallCanvas.width = 100;
        smallCanvas.height = 100;
        const smallCtx = smallCanvas.getContext("2d");
        let score = 0;
        if (smallCtx) {
          smallCtx.drawImage(video, 0, 0, 100, 100);
          const pixels = smallCtx.getImageData(0, 0, 100, 100).data;
          let diff = 0;
          for (let p = 0; p < pixels.length - 4; p += 4) {
            diff += Math.abs(pixels[p] - pixels[p+4]);
          }
          score = diff;
        }

        frames.push({ 
          data: fullCanvas.toDataURL("image/jpeg", 0.9), 
          score 
        });
      }
      await new Promise(r => setTimeout(r, 100));
    }

    // Step 2: Select Sharpest Frame
    const bestFrame = frames.sort((a, b) => b.score - a.score)[0];
    onStableFrame(bestFrame.data);
  };

  const handleButtonClick = () => {
    if (isProcessing) {
      if (onStop) onStop();
      return;
    }
    triggerManualCapture();
  };

  useEffect(() => {
    setHasMounted(true);
    if (!shouldEnable) return;

    // Start camera
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: "environment", width: 1280, height: 720 } })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch((err) => console.error("Error accessing camera:", err));

    let animationFrameId: number;

    const processFrame = () => {
      if (!videoRef.current || !canvasRef.current || !shouldEnable) {
        animationFrameId = requestAnimationFrame(processFrame);
        return;
      }

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });

      if (!ctx || video.videoWidth === 0) {
        animationFrameId = requestAnimationFrame(processFrame);
        return;
      }

      // Grayscale/Motion logic remains for the HUD metrics
      canvas.width = 64;
      canvas.height = 64;
      ctx.drawImage(video, 0, 0, 64, 64);
      const frameData = ctx.getImageData(0, 0, 64, 64).data;
      const currentGray = new Uint8ClampedArray(64 * 64);
      for (let i = 0, j = 0; i < frameData.length; i += 4, j++) {
        currentGray[j] = (frameData[i] + frameData[i + 1] + frameData[i + 2]) / 3;
      }

      const { prevGrayData, recentMotions } = stateRef.current;
      let motion = 0;
      if (prevGrayData) {
        let totalDiff = 0;
        for (let i = 0; i < currentGray.length; i++) {
          totalDiff += Math.abs(currentGray[i] - prevGrayData[i]);
        }
        motion = totalDiff / currentGray.length;
      }
      stateRef.current.prevGrayData = currentGray;
      recentMotions.push(motion);
      if (recentMotions.length > 10) recentMotions.shift();
      
      if (Math.random() < 0.2) setMotionScore(motion);
      
      const avgMotion = recentMotions.reduce((a, b) => a + b, 0) / recentMotions.length;
      if (avgMotion < 4.0) setStatus("STABLE");
      else if (avgMotion > 8.0) setStatus("MOTION_DETECTED");
      else setStatus("IDLE");

      setTimeout(() => {
        animationFrameId = requestAnimationFrame(processFrame);
      }, 100);
    };

    animationFrameId = requestAnimationFrame(processFrame);

    return () => {
      cancelAnimationFrame(animationFrameId);
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [shouldEnable]);

  return (
    <div className="camera-container">
      {hasMounted ? (
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="live-video"
          suppressHydrationWarning
        />
      ) : (
        <div className="live-video-placeholder" />
      )}
      <canvas ref={canvasRef} style={{ display: "none" }} />
      
      <div className="hud-overlay">
        <div className="hud-metric">Stability: {status}</div>
        {isProcessing && <div className="hud-status status-analyzing">AI IS WORKING...</div>}
      </div>

      {shouldEnable && (
        <div className={`capture-controls ${isProcessing ? "active" : ""}`}>
          <button 
            className={`shutter-button ${isProcessing ? "stop-mode" : ""} ${status === "STABLE" && !isProcessing ? "ready" : ""}`}
            onClick={handleButtonClick}
            aria-label={isProcessing ? "Stop AI" : "Capture Question"}
          >
            <div className="shutter-inner"></div>
          </button>
        </div>
      )}
    </div>
  );
}
