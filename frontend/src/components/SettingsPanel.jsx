import "./SettingsPanel.css";

const SNOOZE_OPTIONS = [
  { label: "5 min", seconds: 5 * 60 },
  { label: "15 min", seconds: 15 * 60 },
  { label: "30 min", seconds: 30 * 60 },
  { label: "Until next break", seconds: null }, // handled specially by caller
];

export default function SettingsPanel({
  alertKinds,
  onToggleKind,
  sedentaryThresholdMin,
  onChangeSedentaryThreshold,
  onSnooze,
  snoozeRemaining,
  disabled,
}) {
  return (
    <div className="settings-panel">
      <div className="settings-block">
        <h3>Alert channels</h3>
        <p className="settings-hint">Choose how you want to be nudged. Pick any combination.</p>
        <div className="toggle-row">
          {[
            { key: "sound", label: "Sound" },
            { key: "popup", label: "Browser popup" },
            { key: "text", label: "On-screen text" },
          ].map(({ key, label }) => (
            <button
              key={key}
              type="button"
              className={`toggle-chip ${alertKinds.has(key) ? "on" : ""}`}
              onClick={() => onToggleKind(key)}
              aria-pressed={alertKinds.has(key)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="settings-block">
        <h3>Sedentary threshold</h3>
        <p className="settings-hint">First gentle reminder after this many minutes seated.</p>
        <input
          type="range"
          min="10"
          max="120"
          step="10"
          value={sedentaryThresholdMin}
          onChange={(e) => onChangeSedentaryThreshold(Number(e.target.value))}
          disabled={disabled}
        />
        <div className="range-value">{sedentaryThresholdMin} min</div>
      </div>

      <div className="settings-block">
        <h3>Snooze</h3>
        <p className="settings-hint">
          {snoozeRemaining > 0
            ? `Alerts snoozed: ${Math.floor(snoozeRemaining / 60)}:${String(
                snoozeRemaining % 60
              ).padStart(2, "0")}`
            : "Pause alerts temporarily."}
        </p>
        <div className="toggle-row">
          {SNOOZE_OPTIONS.map((opt) => (
            <button
              key={opt.label}
              type="button"
              className="toggle-chip"
              onClick={() => onSnooze(opt.seconds)}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
