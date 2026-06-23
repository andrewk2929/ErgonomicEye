// Posture geometry helpers.
//
// All functions take MediaPipe-style landmarks: { x, y } normalized 0-1
// (we don't need z or pixel coords for these particular angles).

/** Angle ABC in degrees, at vertex b. */
export function calculateAngle(a, b, c) {
  const radians =
    Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let angle = Math.abs((radians * 180.0) / Math.PI);
  if (angle > 180.0) angle = 360 - angle;
  return angle;
}

export function midpoint(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

/**
 * Extracts the angles we care about from a single frame of pose landmarks.
 * MediaPipe Pose landmark indices: 11 = left shoulder, 12 = right shoulder,
 * 0 = nose, 23 = left hip, 24 = right hip.
 */
export function extractPostureAngles(landmarks) {
  const leftShoulder = landmarks[11];
  const rightShoulder = landmarks[12];
  const nose = landmarks[0];
  const leftHip = landmarks[23];
  const rightHip = landmarks[24];

  const midShoulder = midpoint(leftShoulder, rightShoulder);
  const midHip = midpoint(leftHip, rightHip);

  // Shoulder "openness" angle as seen by the camera (nose as vertex
  // between the two shoulders) - shrinks as shoulders round forward.
  const shoulderAngle = 180 - calculateAngle(leftShoulder, nose, rightShoulder);

  // Neck angle: how far the head tilts forward/down relative to a
  // straight-up line above the shoulder midpoint. A point directly
  // above midShoulder represents "perfectly upright".
  const neckAngle = calculateAngle(midShoulder, nose, {
    x: midShoulder.x,
    y: midShoulder.y - 1,
  });

  // Torso angle: spine lean, using hip midpoint -> shoulder midpoint
  // versus straight-up. Lower angle = more slouched/leaning.
  const torsoAngle = calculateAngle(midHip, midShoulder, {
    x: midShoulder.x,
    y: midShoulder.y - 1,
  });

  // Shoulder slope: positive = right shoulder lower than left (image
  // coordinates have y growing downward), used to flag uneven shoulders.
  const dx = rightShoulder.x - leftShoulder.x;
  const shoulderSlope = dx !== 0 ? (rightShoulder.y - leftShoulder.y) / dx : 0;

  return { neckAngle, shoulderAngle, torsoAngle, shoulderSlope };
}

/**
 * Severity scoring, mirrors backend/posture_logic.py exactly so the
 * on-screen status never disagrees with what gets logged. Each of the
 * 3 dimensions contributes 0-1 (capped), summed into 0-3 severity.
 */
export function scoreDimension(current, baseline, tolerance = 10) {
  const diff = Math.abs(current - baseline);
  return Math.min(diff / tolerance, 1.0);
}

export function computeSeverity(angles, baseline) {
  const neckScore = scoreDimension(angles.neckAngle, baseline.neckAngle);
  const shoulderScore = scoreDimension(angles.shoulderAngle, baseline.shoulderAngle);
  const torsoScore = scoreDimension(angles.torsoAngle, baseline.torsoAngle);

  const severity = neckScore + shoulderScore + torsoScore;

  let status = "good";
  if (severity >= 1.5) status = "poor";
  else if (severity >= 0.6) status = "warning";

  let issue = null;
  if (status !== "good") {
    const scores = {
      forward_head: neckScore,
      uneven_shoulders: shoulderScore,
      slouching: torsoScore,
    };
    issue = Object.keys(scores).reduce((a, b) => (scores[a] > scores[b] ? a : b));
  }

  return { severity, status, issue, neckScore, shoulderScore, torsoScore };
}

export const ISSUE_COPY = {
  forward_head: {
    label: "Forward head posture",
    coachTip:
      "Your head is drifting forward of your shoulders. Try sliding your monitor back a few inches, or raising it 2-3 inches so your eyes meet the top third of the screen.",
  },
  uneven_shoulders: {
    label: "Uneven shoulders",
    coachTip:
      "One shoulder is sitting higher than the other. Check whether your mouse, chair armrest, or desk setup favors one side, and try squaring your hips to the desk.",
  },
  slouching: {
    label: "Slouching",
    coachTip:
      "Your torso has leaned away from your calibrated upright position. Try sitting back so your lower back meets the chair, or add lumbar support.",
  },
};

export const SEDENTARY_TIERS = [
  { seconds: 30 * 60, level: "gentle", message: "You've been seated for 30 minutes. Try a quick shoulder stretch." },
  { seconds: 60 * 60, level: "walk", message: "You've been seated for an hour. Take a 2 minute walk." },
  { seconds: 90 * 60, level: "strong", message: "You've been seated for 90 minutes. Please take a real break." },
];

export function getSedentaryTier(secondsSeated) {
  let tier = null;
  for (const t of SEDENTARY_TIERS) {
    if (secondsSeated >= t.seconds) tier = t;
  }
  return tier;
}
