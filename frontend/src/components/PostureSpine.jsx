import "./PostureSpine.css";

const STATUS_COLOR = {
  idle: "var(--text-faint)",
  good: "var(--sage)",
  warning: "var(--gold)",
  poor: "var(--clay)",
};

/**
 * The signature element: a vertical "spine" that tilts in real time
 * with the user's actual neck/torso angle deviation from baseline, and
 * recolors by status. Not a generic gauge or progress ring - it's a
 * literal small drawing of the thing being measured.
 */
export default function PostureSpine({ status = "idle", tiltDegrees = 0, label }) {
  const color = STATUS_COLOR[status] || STATUS_COLOR.idle;
  // Clamp visual tilt so it stays legible even with big angle deviations.
  const clamped = Math.max(-22, Math.min(22, tiltDegrees));

  return (
    <div className="spine-wrap">
      <svg viewBox="0 0 120 160" className="spine-svg" aria-hidden="true">
        <line x1="60" y1="8" x2="60" y2="152" className="spine-guide" />
        <g style={{ transform: `rotate(${clamped}deg)`, transformOrigin: "60px 150px" }}>
          <circle cx="60" cy="22" r="14" fill={color} className="spine-head" />
          <line
            x1="60"
            y1="36"
            x2="60"
            y2="150"
            stroke={color}
            strokeWidth="6"
            strokeLinecap="round"
            className="spine-body"
          />
          <line x1="38" y1="56" x2="82" y2="56" stroke={color} strokeWidth="6" strokeLinecap="round" />
        </g>
      </svg>
      <span className="spine-label" style={{ color }}>
        {label}
      </span>
    </div>
  );
}
