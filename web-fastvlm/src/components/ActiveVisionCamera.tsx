"use client";

import React, { useEffect, useRef, useState } from "react";

interface ActiveVisionCameraProps {
  onStableFrame: (base64Image: string) => void;
  isProcessing: boolean;
}

export default function ActiveVisionCamera({ onStableFrame, isProcessing }: ActiveVisionCameraProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [motionScore, setMotionScore] = useState<number>(0);
  const [status, setStatus] = useState<"IDLE" | "MOTION_DETECTED" | "STABLE">("IDLE");
  const [hasMounted, setHasMounted] = useState(false);

  // Internal refs to track stability across frames without triggering re-renders for every frame
  const stateRef = useRef({
    prevGrayData: null as Uint8ClampedArray | null,
    recentMotions: [] as number[],
    hasNewContent: false,
    lastInferenceTime: 0,
  });

  const MOTION_BASELINE = 8.0;
  const STABLE_MAX_MOTION = 4.5;
  const HISTORY_LENGTH = 8;
  const COOLDOWN_MS = 5000;

  useEffect(() => {
    setHasMounted(true);
    // Start camera
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: "environment" } })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch((err) => console.error("Error accessing camera:", err));

    let animationFrameId: number;

    const processFrame = () => {
      if (!videoRef.current || !canvasRef.current || isProcessing) {
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

      // Small resolution for motion calculation (64x64)
      canvas.width = 64;
      canvas.height = 64;
      ctx.drawImage(video, 0, 0, 64, 64);

      const frameData = ctx.getImageData(0, 0, 64, 64).data;
      const currentGray = new Uint8ClampedArray(64 * 64);

      for (let i = 0, j = 0; i < frameData.length; i += 4, j++) {
        // R, G, B to grayscale
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
      
      // Update rolling history
      recentMotions.push(motion);
      if (recentMotions.length > HISTORY_LENGTH) {
        recentMotions.shift();
      }

      // Only update UI state occasionally to prevent stutter
      if (Math.random() < 0.2) setMotionScore(motion);

      const now = Date.now();
      if (now - stateRef.current.lastInferenceTime > COOLDOWN_MS && !isProcessing) {
        if (motion > MOTION_BASELINE) {
          stateRef.current.hasNewContent = true;
          setStatus("MOTION_DETECTED");
        }

        const avgMotion = recentMotions.length === HISTORY_LENGTH 
          ? recentMotions.reduce((a, b) => a + b, 0) / HISTORY_LENGTH 
          : Infinity;

        if (stateRef.current.hasNewContent && avgMotion < STABLE_MAX_MOTION) {
          setStatus("STABLE");
          stateRef.current.hasNewContent = false;
          stateRef.current.lastInferenceTime = now;
          
          // Capture full resolution frame for AI
          const fullCanvas = document.createElement("canvas");
          fullCanvas.width = video.videoWidth;
          fullCanvas.height = video.videoHeight;
          const fullCtx = fullCanvas.getContext("2d");
          if (fullCtx) {
            fullCtx.drawImage(video, 0, 0);
            const base64Image = fullCanvas.toDataURL("image/jpeg", 0.8);
            onStableFrame(base64Image);
          }
        } else if (!stateRef.current.hasNewContent) {
          setStatus("IDLE");
        }
      }

      // Schedule next check (~10 fps is enough for motion gating)
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
  }, [isProcessing, onStableFrame]);

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
        <div className="hud-metric">Motion: {motionScore.toFixed(1)}</div>
        <div className={`hud-status status-${status.toLowerCase()}`}>
          {isProcessing ? "ANALYZING..." : status}
        </div>
      </div>
    </div>
  );
}
