import "./AlertBanner.css";

export default function AlertBanner({ alert, onDismiss }) {
  if (!alert) return null;

  return (
    <div className={`alert-banner alert-${alert.level}`} role="alert">
      <div className="alert-banner-text">
        <strong>{alert.title}</strong>
        <span>{alert.message}</span>
      </div>
      <button type="button" className="alert-dismiss" onClick={onDismiss} aria-label="Dismiss">
        ×
      </button>
    </div>
  );
}
