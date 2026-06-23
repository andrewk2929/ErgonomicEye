const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${path} failed: ${res.status} ${body}`);
  }
  return res.json();
}

export const api = {
  createCalibration: (payload) =>
    request("/api/calibration", { method: "POST", body: JSON.stringify(payload) }),

  getLatestCalibration: (sessionId) =>
    request(`/api/calibration/${sessionId}/latest`),

  logPostureEvent: (payload) =>
    request("/api/posture-event", { method: "POST", body: JSON.stringify(payload) }),

  logAlert: (payload) =>
    request("/api/alert", { method: "POST", body: JSON.stringify(payload) }),

  startSedentarySession: (sessionId) =>
    request("/api/sedentary/start", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId }),
    }),

  endSedentarySession: (sessionId, sedentarySessionDbId) =>
    request("/api/sedentary/end", {
      method: "POST",
      body: JSON.stringify({
        session_id: sessionId,
        sedentary_session_db_id: sedentarySessionDbId,
      }),
    }),

  getDailyAnalytics: (sessionId) => request(`/api/analytics/daily/${sessionId}`),

  getWeeklyAnalytics: (sessionId) => request(`/api/analytics/weekly/${sessionId}`),
};
