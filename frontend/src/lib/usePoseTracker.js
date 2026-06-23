import { useEffect, useRef, useState, useCallback } from "react";
import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "@mediapipe/tasks-vision";

/**
 * Runs MediaPipe's Pose Landmarker entirely client-side against the
 * visitor's own webcam. This is the fix for the original app: a
 * deployed website can never reach into a visitor's machine and open
 * cv2.VideoCapture(0) on a server - the camera and the model both have
 * to live in the browser that owns the camera.
 */
export function usePoseTracker({ videoRef, canvasRef, onLandmarks, active }) {
  const landmarkerRef = useRef(null);
  const rafRef = useRef(null);
  const drawingUtilsRef = useRef(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState(null);
  const [cameraGranted, setCameraGranted] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        );
        const landmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numPoses: 1,
        });
        if (cancelled) return;
        landmarkerRef.current = landmarker;
        setReady(true);
      } catch (e) {
        console.error("Failed to load pose landmarker:", e);
        if (!cancelled) setError("Could not load the pose detection model. Check your connection and reload.");
      }
    }

    init();
    return () => {
      cancelled = true;
      landmarkerRef.current?.close();
    };
  }, []);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraGranted(true);
      setError(null);
    } catch (e) {
      console.error("Camera access failed:", e);
      setError(
        "Camera access was blocked or unavailable. Allow camera permissions for this site and try again."
      );
    }
  }, [videoRef]);

  const stopCamera = useCallback(() => {
    const video = videoRef.current;
    if (video && video.srcObject) {
      video.srcObject.getTracks().forEach((t) => t.stop());
      video.srcObject = null;
    }
    setCameraGranted(false);
  }, [videoRef]);

  useEffect(() => {
    if (!active || !ready || !cameraGranted) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (canvas && ctx && !drawingUtilsRef.current) {
      drawingUtilsRef.current = new DrawingUtils(ctx);
    }

    let lastVideoTime = -1;

    function loop() {
      const landmarker = landmarkerRef.current;
      if (video && landmarker && video.readyState >= 2) {
        if (video.currentTime !== lastVideoTime) {
          lastVideoTime = video.currentTime;
          const result = landmarker.detectForVideo(video, performance.now());

          if (canvas && ctx) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.save();
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            if (result.landmarks?.[0]) {
              const du = drawingUtilsRef.current;
              du.drawLandmarks(result.landmarks[0], { radius: 3 });
              du.drawConnectors(result.landmarks[0], PoseLandmarker.POSE_CONNECTIONS);
            }
            ctx.restore();
          }

          if (result.landmarks?.[0]) {
            onLandmarks(result.landmarks[0]);
          } else {
            onLandmarks(null);
          }
        }
      }
      rafRef.current = requestAnimationFrame(loop);
    }

    rafRef.current = requestAnimationFrame(loop);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [active, ready, cameraGranted, videoRef, canvasRef, onLandmarks]);

  return { ready, error, cameraGranted, startCamera, stopCamera };
}
