"""
Posture scoring logic.

This mirrors what the frontend computes live (it has to - the frontend
needs an instant on-screen status without waiting on a network
round-trip), but the backend recomputes from raw angles before storing
so the database is the source of truth and can't be spoofed by a
tampered client.
"""


def score_dimension(current, baseline, tolerance=10):
    """
    Returns 0-1 severity for a single angle dimension.
    0 = matches baseline, 1 = off by >= tolerance degrees.
    """
    diff = abs(current - baseline)
    return min(diff / tolerance, 1.0)


def compute_severity(neck_angle, shoulder_angle, torso_angle, baseline):
    """
    severity = neck_score + shoulder_score + torso_score   (per the outline)
    Each term is 0-1, so total severity is 0-3.
    """
    neck_score = score_dimension(neck_angle, baseline["neck_angle"])
    shoulder_score = score_dimension(shoulder_angle, baseline["shoulder_angle"])
    torso_score = score_dimension(torso_angle, baseline["torso_angle"])

    severity = neck_score + shoulder_score + torso_score

    if severity < 0.6:
        status = "good"
        issue = None
    elif severity < 1.5:
        status = "warning"
        issue = _dominant_issue(neck_score, shoulder_score, torso_score)
    else:
        status = "poor"
        issue = _dominant_issue(neck_score, shoulder_score, torso_score)

    return {
        "severity": round(severity, 3),
        "status": status,
        "issue": issue,
        "neck_score": round(neck_score, 3),
        "shoulder_score": round(shoulder_score, 3),
        "torso_score": round(torso_score, 3),
    }


def _dominant_issue(neck_score, shoulder_score, torso_score):
    scores = {
        "forward_head": neck_score,
        "uneven_shoulders": shoulder_score,
        "slouching": torso_score,
    }
    return max(scores, key=scores.get)


SEDENTARY_TIERS = [
    (30 * 60, "gentle", "You've been seated for 30 minutes. Try a quick shoulder stretch."),
    (60 * 60, "walk", "You've been seated for an hour. Take a 2 minute walk."),
    (90 * 60, "strong", "You've been seated for 90 minutes. Please take a real break."),
]


def sedentary_tier(seconds_seated):
    """Returns the tier dict that applies right now, or None if under 30 min."""
    tier = None
    for threshold, level, message in SEDENTARY_TIERS:
        if seconds_seated >= threshold:
            tier = {"level": level, "message": message, "threshold": threshold}
    return tier
