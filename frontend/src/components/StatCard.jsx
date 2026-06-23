import "./StatCard.css";

export default function StatCard({ label, value, unit, accent }) {
  return (
    <div className="stat-card">
      <span className="stat-label">{label}</span>
      <span className="stat-value" style={accent ? { color: `var(--${accent})` } : undefined}>
        {value}
        {unit && <span className="stat-unit">{unit}</span>}
      </span>
    </div>
  );
}
