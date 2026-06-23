"""
ErgonomicEye backend.

What this service is and isn't responsible for:
- IS responsible for: storing calibration baselines, logging posture
  events/alerts, computing analytics, deciding sedentary tiers.
- IS NOT responsible for: webcam access or running MediaPipe. That all
  happens client-side in the browser (see frontend/). A server can't
  see a visitor's webcam, so pose detection has to run on their machine.

Run with: uvicorn main:app --reload --port 8000
"""
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel
from typing import Optional, Literal
import uuid

from database import init_db, get_db
from models import Calibration, PostureEvent, AlertLog, SedentarySession
from posture_logic import compute_severity, sedentary_tier

app = FastAPI(title="ErgonomicEye API")

# Wide-open CORS because the frontend can be deployed to any static host
# (Vercel, Netlify, etc.) and we don't know that origin ahead of time.
# Tighten this to your actual frontend domain before going to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


# ---------- Schemas ----------

class CalibrationIn(BaseModel):
    session_id: Optional[str] = None
    neck_angle: float
    shoulder_angle: float
    torso_angle: float
    shoulder_slope: float


class PostureEventIn(BaseModel):
    session_id: str
    neck_angle: float
    shoulder_angle: float
    torso_angle: float


class AlertIn(BaseModel):
    session_id: str
    alert_kind: Literal["posture", "sedentary"]
    issue: Optional[str] = None
    message: str


class SedentaryStartIn(BaseModel):
    session_id: str


class SedentaryEndIn(BaseModel):
    session_id: str
    sedentary_session_db_id: int


# ---------- Calibration ----------

@app.post("/api/calibration")
def create_calibration(payload: CalibrationIn, db: Session = Depends(get_db)):
    """
    Stores a new personal baseline. Called once after the frontend's
    25-frame (~5-10s) calibration window finishes averaging angles.
    """
    session_id = payload.session_id or str(uuid.uuid4())
    cal = Calibration(
        session_id=session_id,
        neck_angle=payload.neck_angle,
        shoulder_angle=payload.shoulder_angle,
        torso_angle=payload.torso_angle,
        shoulder_slope=payload.shoulder_slope,
    )
    db.add(cal)
    db.commit()
    db.refresh(cal)
    return {"session_id": session_id, "calibration_id": cal.id}


@app.get("/api/calibration/{session_id}/latest")
def get_latest_calibration(session_id: str, db: Session = Depends(get_db)):
    cal = (
        db.query(Calibration)
        .filter(Calibration.session_id == session_id)
        .order_by(Calibration.created_at.desc())
        .first()
    )
    if not cal:
        raise HTTPException(status_code=404, detail="No calibration found for this session")
    return {
        "neck_angle": cal.neck_angle,
        "shoulder_angle": cal.shoulder_angle,
        "torso_angle": cal.torso_angle,
        "shoulder_slope": cal.shoulder_slope,
    }


# ---------- Posture events ----------

@app.post("/api/posture-event")
def log_posture_event(payload: PostureEventIn, db: Session = Depends(get_db)):
    """
    The frontend polls this every few seconds (not every frame) with raw
    angles. The backend recomputes severity server-side against the
    stored baseline, logs the event, and hands back the verdict so the
    frontend's alert state and the DB never disagree.
    """
    cal = (
        db.query(Calibration)
        .filter(Calibration.session_id == payload.session_id)
        .order_by(Calibration.created_at.desc())
        .first()
    )
    if not cal:
        raise HTTPException(status_code=404, detail="Calibrate first")

    baseline = {
        "neck_angle": cal.neck_angle,
        "shoulder_angle": cal.shoulder_angle,
        "torso_angle": cal.torso_angle,
    }
    result = compute_severity(
        payload.neck_angle, payload.shoulder_angle, payload.torso_angle, baseline
    )

    event = PostureEvent(
        session_id=payload.session_id,
        neck_angle=payload.neck_angle,
        shoulder_angle=payload.shoulder_angle,
        torso_angle=payload.torso_angle,
        severity=result["severity"],
        status=result["status"],
        issue=result["issue"],
    )
    db.add(event)
    db.commit()

    return result


@app.post("/api/alert")
def log_alert(payload: AlertIn, db: Session = Depends(get_db)):
    """
    The frontend calls this only when it actually fires an alert (i.e.
    after its own 30-second sustained-issue timer elapses, and respecting
    snooze) - so this table is a true count of interventions, not raw
    posture samples.
    """
    log = AlertLog(
        session_id=payload.session_id,
        alert_kind=payload.alert_kind,
        issue=payload.issue,
        message=payload.message,
    )
    db.add(log)
    db.commit()
    return {"ok": True, "id": log.id}


# ---------- Sedentary session tracking ----------

@app.post("/api/sedentary/start")
def start_sedentary_session(payload: SedentaryStartIn, db: Session = Depends(get_db)):
    s = SedentarySession(session_id=payload.session_id)
    db.add(s)
    db.commit()
    db.refresh(s)
    return {"sedentary_session_db_id": s.id}


@app.post("/api/sedentary/end")
def end_sedentary_session(payload: SedentaryEndIn, db: Session = Depends(get_db)):
    s = db.query(SedentarySession).filter(SedentarySession.id == payload.sedentary_session_db_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Sedentary session not found")
    s.ended_at = datetime.now(timezone.utc)
    started = s.started_at
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    s.duration_seconds = (s.ended_at - started).total_seconds()
    db.commit()
    return {"duration_seconds": s.duration_seconds}


@app.get("/api/sedentary/tier")
def get_sedentary_tier(seconds_seated: float):
    """Stateless helper - frontend can call this with its local timer value."""
    tier = sedentary_tier(seconds_seated)
    return {"tier": tier}


# ---------- Analytics dashboard ----------

@app.get("/api/analytics/daily/{session_id}")
def daily_analytics(session_id: str, db: Session = Depends(get_db)):
    """
    Today's: average posture score, longest sitting session, number of
    alerts, total time spent sitting. (per the outline's dashboard spec)
    """
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    events_today = (
        db.query(PostureEvent)
        .filter(PostureEvent.session_id == session_id, PostureEvent.created_at >= today_start)
        .all()
    )
    avg_severity = (
        sum(e.severity for e in events_today) / len(events_today) if events_today else 0
    )
    # turn severity (0-3, lower better) into a 0-100 "posture score" (higher better)
    avg_posture_score = round(max(0, 100 - (avg_severity / 3) * 100), 1)

    alert_count = (
        db.query(func.count(AlertLog.id))
        .filter(AlertLog.session_id == session_id, AlertLog.created_at >= today_start)
        .scalar()
    )

    sedentary_sessions = (
        db.query(SedentarySession)
        .filter(SedentarySession.session_id == session_id, SedentarySession.started_at >= today_start)
        .all()
    )
    longest_session = max((s.duration_seconds or 0 for s in sedentary_sessions), default=0)
    total_sitting = sum((s.duration_seconds or 0 for s in sedentary_sessions))

    return {
        "average_posture_score": avg_posture_score,
        "longest_sitting_session_seconds": longest_session,
        "alert_count": alert_count or 0,
        "total_sitting_seconds": total_sitting,
        "samples": len(events_today),
    }


@app.get("/api/analytics/weekly/{session_id}")
def weekly_analytics(session_id: str, db: Session = Depends(get_db)):
    """
    Posture trend + improvement % over the last 7 days, bucketed by day,
    so the frontend can plot a simple line chart.
    """
    week_start = datetime.now(timezone.utc) - timedelta(days=7)

    events = (
        db.query(PostureEvent)
        .filter(PostureEvent.session_id == session_id, PostureEvent.created_at >= week_start)
        .order_by(PostureEvent.created_at.asc())
        .all()
    )

    buckets = {}
    for e in events:
        day_key = e.created_at.strftime("%Y-%m-%d")
        buckets.setdefault(day_key, []).append(e.severity)

    trend = []
    for day, severities in sorted(buckets.items()):
        avg = sum(severities) / len(severities)
        score = round(max(0, 100 - (avg / 3) * 100), 1)
        trend.append({"date": day, "posture_score": score})

    improvement_pct = None
    if len(trend) >= 2:
        first, last = trend[0]["posture_score"], trend[-1]["posture_score"]
        if first > 0:
            improvement_pct = round(((last - first) / first) * 100, 1)

    return {"trend": trend, "improvement_pct": improvement_pct}


@app.get("/api/health")
def health():
    return {"status": "ok"}
