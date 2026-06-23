"""
Database models for ErgonomicEye.

SQLite for now (per the project outline) - swapping to Postgres later
just means changing the DATABASE_URL in database.py, the models below
don't need to change.
"""
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base
from datetime import datetime, timezone

Base = declarative_base()


def utcnow():
    return datetime.now(timezone.utc)


class Calibration(Base):
    """
    A user's personal posture baseline, captured during the
    5-10 second calibration step. Nothing is hardcoded - every
    session computes its own baseline because everyone sits
    differently (per the outline).
    """
    __tablename__ = "calibrations"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    neck_angle = Column(Float)
    shoulder_angle = Column(Float)
    torso_angle = Column(Float)
    shoulder_slope = Column(Float)
    created_at = Column(DateTime, default=utcnow)


class PostureEvent(Base):
    """
    A single posture reading, logged periodically (not every frame -
    that would flood the DB). Used to build the analytics dashboard
    and to detect sustained (not momentary) posture problems.
    """
    __tablename__ = "posture_events"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    neck_angle = Column(Float)
    shoulder_angle = Column(Float)
    torso_angle = Column(Float)
    severity = Column(Float)          # 0 = perfect, higher = worse
    status = Column(String)           # "good" | "warning" | "poor"
    issue = Column(String, nullable=True)  # "forward_head" | "uneven_shoulders" | "slouching" | None
    created_at = Column(DateTime, default=utcnow)


class AlertLog(Base):
    """
    Every time the user was actually nudged (sound/popup/text), so the
    dashboard can show "X alerts today" and the pattern-discovery layer
    can later find things like "most alerts happen 2-4pm".
    """
    __tablename__ = "alert_log"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    alert_kind = Column(String)       # "posture" | "sedentary"
    issue = Column(String, nullable=True)
    message = Column(String)
    created_at = Column(DateTime, default=utcnow)


class SedentarySession(Base):
    """
    Tracks one continuous sitting session, closed out when the user
    gets up (no pose detected for a while) or stops the app. Powers
    "longest sitting session" and total sitting time in the dashboard.
    """
    __tablename__ = "sedentary_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    started_at = Column(DateTime, default=utcnow)
    ended_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)
