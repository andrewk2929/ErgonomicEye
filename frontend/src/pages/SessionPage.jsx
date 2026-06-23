import { useRef, useState, useCallback, useEffect } from "react";
import { usePoseTracker } from "../lib/usePoseTracker";
import {
  extractPostureAngles,
  computeSeverity,
  ISSUE_COPY,
} from "../lib/posture";
import { fireAlert, ensureNotificationPermission } from "../lib/alerts";
import { api } from "../lib/api";
import PostureSpine from "../components/PostureSpine";
import SettingsPanel from "../components/SettingsPanel";
import AlertBanner from "../components/AlertBanner";
import StatCard from "../components/StatCard";
import "./SessionPage.css";

const CALIBRATION_FRAMES = 60; // ~2s of frames at typical webcam rate via rAF
const SUSTAIN_SECONDS = 30; // outline spec: only alert after 30s sustained issue
const ALERT_COOLDOWN_SECONDS = 30;
const LOG_INTERVAL_SECONDS = 5; // how often we send a posture-event to the backend

function getOrCreateSessionId() {
  let id = localStorage.getItem("ee_session_id");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("ee_session_id", id);
  }
  return id;
}

export default function SessionPage() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const sessionId = useRef(getOrCreateSessionId()).current;

  const [phase, setPhase] = useState("idle"); // idle | calibrating | live
  const [calibrationProgress, setCalibrationProgress] = useState(0);
  const [baseline, setBaseline] = useState(null);
  const [liveResult, setLiveResult] = useState(null);
  const [noPersonVisible, setNoPersonVisible] = useState(false);
  const [banner, setBanner] = useState(null);
  const [error, setError] = useState(null);

  const [alertKinds, setAlertKinds] = useState(new Set(["sound", "popup", "text"]));
  const [sedentaryThresholdMin, setSedentaryThresholdMin] = useState(60);
  const [snoozeUntil, setSnoozeUntil] = useState(0);
  const [snoozeRemaining, setSnoozeRemaining] = useState(0);

  const [dailyStats, setDailyStats] = useState(null);

  // Refs for values read inside the rAF loop's callback (avoids stale
  // closures without re-subscribing the pose tracker every render).
  const calibSamplesRef = useRef([]);
  const issueStartRef = useRef({}); // { forward_head: timestamp, ... }
  const lastAlertAtRef = useRef(0);
  const lastLogAtRef = useRef(0);
  const sessionStartRef = useRef(null);
  const sedentaryDbIdRef = useRef(null);
  const sedentaryAlertedTiersRef = useRef(new Set());
  const alertKindsRef = useRef(alertKinds);
  const snoozeUntilRef = useRef(0);
  const sedentaryThresholdMinRef = useRef(sedentaryThresholdMin);

  useEffect(() => {
    alertKindsRef.current = alertKinds;
  }, [alertKinds]);
  useEffect(() => {
    snoozeUntilRef.current = snoozeUntil;
  }, [snoozeUntil]);
  useEffect(() => {
    sedentaryThresholdMinRef.current = sedentaryThresholdMin;
  }, [sedentaryThresholdMin]);

  // Snooze countdown ticker
  useEffect(() => {
    const id = setInterval(() => {
      setSnoozeRemaining(Math.max(0, Math.round((snoozeUntil - Date.now()) / 1000)));
    }, 1000);
    return () => clearInterval(id);
  }, [snoozeUntil]);

  const raiseAlert = useCallback(
    (level, issueKey, message, kind = "posture") => {
      const now = Date.now();
      if (now < snoozeUntilRef.current) return;
      if (now - lastAlertAtRef.current < ALERT_COOLDOWN_SECONDS * 1000) return;
      lastAlertAtRef.current = now;

      const title =
        kind === "sedentary"
          ? "Time to move"
          : ISSUE_COPY[issueKey]?.label || "Posture check";

      const showText = fireAlert({ kinds: alertKindsRef.current, title, message });
      if (showText) {
        setBanner({ level, title, message });
      }

      api
        .logAlert({ session_id: sessionId, alert_kind: kind, issue: issueKey, message })
        .catch((e) => console.warn("Failed to log alert:", e));
    },
    [sessionId]
  );

  const handleLandmarks = useCallback(
    (landmarks) => {
      if (!landmarks) {
        setNoPersonVisible(true);
        return;
      }
      setNoPersonVisible(false);

      const angles = extractPostureAngles(landmarks);

      if (phase === "calibrating") {
        calibSamplesRef.current.push(angles);
        setCalibrationProgress(calibSamplesRef.current.length);

        if (calibSamplesRef.current.length >= CALIBRATION_FRAMES) {
          const n = calibSamplesRef.current.length;
          const avg = calibSamplesRef.current.reduce(
            (acc, a) => ({
              neckAngle: acc.neckAngle + a.neckAngle / n,
              shoulderAngle: acc.shoulderAngle + a.shoulderAngle / n,
              torsoAngle: acc.torsoAngle + a.torsoAngle / n,
              shoulderSlope: acc.shoulderSlope + a.shoulderSlope / n,
            }),
            { neckAngle: 0, shoulderAngle: 0, torsoAngle: 0, shoulderSlope: 0 }
          );

          setBaseline(avg);
          setPhase("live");
          sessionStartRef.current = Date.now();
          issueStartRef.current = {};
          sedentaryAlertedTiersRef.current = new Set();

          api
            .createCalibration({
              session_id: sessionId,
              neck_angle: avg.neckAngle,
              shoulder_angle: avg.shoulderAngle,
              torso_angle: avg.torsoAngle,
              shoulder_slope: avg.shoulderSlope,
            })
            .catch((e) => console.warn("Failed to save calibration:", e));

          api
            .startSedentarySession(sessionId)
            .then((res) => {
              sedentaryDbIdRef.current = res.sedentary_session_db_id;
            })
            .catch((e) => console.warn("Failed to start sedentary session:", e));
        }
        return;
      }

      if (phase === "live" && baseline) {
        const result = computeSeverity(angles, baseline);
        setLiveResult({ ...result, angles });

        const now = Date.now();

        // --- Persistence-based posture alerting (only after sustained issue) ---
        if (result.status === "good") {
          issueStartRef.current = {};
        } else if (result.issue) {
          if (!issueStartRef.current[result.issue]) {
            issueStartRef.current[result.issue] = now;
          }
          const sustainedMs = now - issueStartRef.current[result.issue];
          if (sustainedMs >= SUSTAIN_SECONDS * 1000) {
            const copy = ISSUE_COPY[result.issue];
            raiseAlert(
              result.status,
              result.issue,
              copy?.coachTip || "Please check your posture.",
              "posture"
            );
          }
        }

        // --- Sedentary tiers ---
        const secondsSeated = (now - sessionStartRef.current) / 1000;
        // Respect the user's chosen threshold for the first ("gentle")
        // tier; the 60/90 min tiers from the outline scale relative to it.
        const thresholdSec = sedentaryThresholdMinRef.current * 60;
        const tiers = [
          { seconds: thresholdSec, level: "gentle", message: `You've been seated for ${sedentaryThresholdMinRef.current} minutes. Try a quick shoulder stretch.` },
          { seconds: thresholdSec * 2, level: "walk", message: "Take a 2 minute walk to reset." },
          { seconds: thresholdSec * 3, level: "strong", message: "You've been sitting a long time. Please take a real break." },
        ];
        for (const tier of tiers) {
          if (secondsSeated >= tier.seconds && !sedentaryAlertedTiersRef.current.has(tier.level)) {
            sedentaryAlertedTiersRef.current.add(tier.level);
            raiseAlert("warning", null, tier.message, "sedentary");
          }
        }

        // --- Periodic logging to backend ---
        if (now - lastLogAtRef.current >= LOG_INTERVAL_SECONDS * 1000) {
          lastLogAtRef.current = now;
          api
            .logPostureEvent({
              session_id: sessionId,
              neck_angle: angles.neckAngle,
              shoulder_angle: angles.shoulderAngle,
              torso_angle: angles.torsoAngle,
            })
            .catch((e) => console.warn("Failed to log posture event:", e));
        }
      }
    },
    [phase, baseline, raiseAlert, sessionId]
  );

  const { ready, error: trackerError, cameraGranted, startCamera, stopCamera } = usePoseTracker({
    videoRef,
    canvasRef,
    onLandmarks: handleLandmarks,
    active: phase !== "idle",
  });

  const refreshDailyStats = useCallback(() => {
    api
      .getDailyAnalytics(sessionId)
      .then(setDailyStats)
      .catch((e) => console.warn("Failed to load analytics:", e));
  }, [sessionId]);

  useEffect(() => {
    if (phase === "live") {
      const id = setInterval(refreshDailyStats, 15000);
      refreshDailyStats();
      return () => clearInterval(id);
    }
  }, [phase, refreshDailyStats]);

  const handleStart = async () => {
    setError(null);
    await ensureNotificationPermission();
    await startCamera();
    calibSamplesRef.current = [];
    setCalibrationProgress(0);
    setPhase("calibrating");
  };

  const handleStop = () => {
    stopCamera();
    setPhase("idle");
    setBaseline(null);
    setLiveResult(null);
    setBanner(null);
    if (sedentaryDbIdRef.current) {
      api
        .endSedentarySession(sessionId, sedentaryDbIdRef.current)
        .catch((e) => console.warn("Failed to end sedentary session:", e));
      sedentaryDbIdRef.current = null;
    }
    refreshDailyStats();
  };

  const handleSnooze = (seconds) => {
    if (seconds == null) {
      // "Until next break" = snooze for the remaining time until the
      // next sedentary tier would fire.
      const elapsed = sessionStartRef.current ? (Date.now() - sessionStartRef.current) / 1000 : 0;
      const thresholdSec = sedentaryThresholdMinRef.current * 60;
      const nextTierAt = thresholdSec * (Math.floor(elapsed / thresholdSec) + 1);
      seconds = Math.max(60, nextTierAt - elapsed);
    }
    setSnoozeUntil(Date.now() + seconds * 1000);
  };

  const toggleKind = (key) => {
    setAlertKinds((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const statusLabel = {
    idle: "Not started",
    good: "Good posture",
    warning: "Posture slipping",
    poor: "Poor posture",
  }[liveResult?.status || "idle"];

  const tiltDegrees = liveResult
    ? (liveResult.angles.neckAngle - baseline.neckAngle) * -1
    : 0;

  return (
    <div className="session-page">
      <header className="session-header">
        <div>
          <span className="eyebrow">ErgonomicEye</span>
          <h1>Posture session</h1>
        </div>
        <div className="header-actions">
          {phase !== "idle" && (
            <span className={`camera-pill ${cameraGranted ? "on" : ""}`}>
              {cameraGranted ? "Camera connected" : "Requesting camera…"}
            </span>
          )}
          {phase === "idle" ? (
            <button className="btn btn-primary" onClick={handleStart} disabled={!ready}>
              {ready ? "Start" : "Loading model…"}
            </button>
          ) : (
            <button className="btn btn-secondary" onClick={handleStop}>
              Stop
            </button>
          )}
        </div>
      </header>

      {(error || trackerError) && <div className="error-banner">{error || trackerError}</div>}
      <AlertBanner alert={banner} onDismiss={() => setBanner(null)} />

      <div className="session-grid">
        <div className="video-card">
          <video ref={videoRef} className="hidden-video" playsInline muted />
          <canvas ref={canvasRef} className="pose-canvas" />
          {phase === "idle" && (
            <div className="video-overlay">
              <p>Sit naturally in good posture, then hit Start to calibrate your personal baseline.</p>
            </div>
          )}
          {phase === "calibrating" && (
            <div className="video-overlay">
              <p>Hold still — gathering your baseline…</p>
              <div className="progress-track">
                <div
                  className="progress-fill"
                  style={{ width: `${(calibrationProgress / CALIBRATION_FRAMES) * 100}%` }}
                />
              </div>
            </div>
          )}
          {phase === "live" && noPersonVisible && (
            <div className="video-overlay">
              <p>We lost you — step back into frame.</p>
            </div>
          )}
        </div>

        <div className="side-column">
          <div className="status-card">
            <PostureSpine status={liveResult?.status || "idle"} tiltDegrees={tiltDegrees} label={statusLabel} />
            {liveResult && (
              <dl className="angle-readout">
                <div>
                  <dt>Neck</dt>
                  <dd>{liveResult.angles.neckAngle.toFixed(1)}°</dd>
                </div>
                <div>
                  <dt>Shoulder</dt>
                  <dd>{liveResult.angles.shoulderAngle.toFixed(1)}°</dd>
                </div>
                <div>
                  <dt>Torso</dt>
                  <dd>{liveResult.angles.torsoAngle.toFixed(1)}°</dd>
                </div>
              </dl>
            )}
          </div>

          <SettingsPanel
            alertKinds={alertKinds}
            onToggleKind={toggleKind}
            sedentaryThresholdMin={sedentaryThresholdMin}
            onChangeSedentaryThreshold={setSedentaryThresholdMin}
            onSnooze={handleSnooze}
            snoozeRemaining={snoozeRemaining}
          />
        </div>
      </div>

      {dailyStats && (
        <section className="today-stats">
          <h2>Today</h2>
          <div className="stat-row">
            <StatCard label="Posture score" value={dailyStats.average_posture_score} unit="/ 100" accent="sage" />
            <StatCard
              label="Longest sitting session"
              value={Math.round(dailyStats.longest_sitting_session_seconds / 60)}
              unit="min"
            />
            <StatCard label="Alerts" value={dailyStats.alert_count} accent={dailyStats.alert_count > 0 ? "clay" : undefined} />
            <StatCard
              label="Total sitting"
              value={Math.round(dailyStats.total_sitting_seconds / 60)}
              unit="min"
            />
          </div>
        </section>
      )}
    </div>
  );
}
