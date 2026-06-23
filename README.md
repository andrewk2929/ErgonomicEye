# ErgonomicEye 2.0

Real-time posture and sedentary-behavior monitoring that actually works as a
deployed website.

## Why the original didn't deploy

The Streamlit version called `cv2.VideoCapture(0)`, which opens the webcam on
whatever machine is *running the Python process*. On a deployed server,
that's the server, not the visitor — so it could only ever work when you
personally ran it locally. Streamlit's full-script rerun model also fights a
`while True` capture loop.

The fix: pose detection has to run **in the visitor's own browser**, where
the camera actually lives. This rebuild uses MediaPipe's `tasks-vision`
(WASM/JS) build client-side in React, with a small FastAPI + SQLite backend
for calibration, alert history, and analytics — matching the architecture in
your outline.

## Project structure

```
ergonomic-eye/
├── backend/          FastAPI + SQLite
│   ├── main.py        API routes
│   ├── models.py       SQLAlchemy tables
│   ├── posture_logic.py    severity scoring (mirrors frontend logic)
│   ├── database.py
│   └── requirements.txt
└── frontend/         React + Vite
    ├── src/
    │   ├── lib/
    │   │   ├── posture.js        angle math + severity scoring
    │   │   ├── usePoseTracker.js  MediaPipe pose hook (runs in-browser)
    │   │   ├── alerts.js          sound/popup alert delivery
    │   │   └── api.js             backend client
    │   ├── components/    PostureSpine, SettingsPanel, AlertBanner, StatCard
    │   └── pages/SessionPage.jsx  main calibration + live monitoring view
    └── .env.example
```

## Running locally

**Backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend**
```bash
cd frontend
npm install
cp .env.example .env   # points at your backend
npm run dev
```

Open the printed `localhost` URL, allow camera + notification permissions,
sit up straight, and hit **Start**.

## How it works

1. **Calibration** (~2 seconds of frames): averages your neck, shoulder, and
   torso angles into a personal baseline. Nothing is hardcoded — your
   baseline is yours.
2. **Live detection**: every frame computes a severity score
   (`neck_score + shoulder_score + torso_score`, each 0–1) against your
   baseline. An issue only triggers an alert after it's **sustained for 30
   seconds**, so a single head-turn doesn't set off a false alarm.
3. **Alerts**: choose any combination of sound (synthesized chime, no
   external mp3 file to break), browser popup notification, or an on-screen
   text banner. A 30-second cooldown prevents alert spam.
4. **Sedentary tiers**: gentle reminder at your chosen threshold, a walk
   reminder at 2x that, and a stronger nudge at 3x — scaled off the slider
   instead of fixed at 30/60/90 like a one-size-fits-all default.
5. **Snooze**: 5 / 15 / 30 minutes, or "until next break" (computed against
   your sedentary threshold).
6. **Dashboard**: today's average posture score, longest sitting session,
   alert count, and total sitting time, refreshed every 15s. Weekly trend +
   improvement % is available via `/api/analytics/weekly/{session_id}`
   (not yet wired into the UI — see Next steps).

## Deploying

- **Frontend**: any static host (Vercel, Netlify, Cloudflare Pages). Set
  `VITE_API_BASE` to your deployed backend URL at build time.
- **Backend**: any host that runs a Python process (Render, Railway, Fly.io,
  a VPS). Swap SQLite for Postgres by changing `DATABASE_URL` in
  `database.py` — no model changes needed. Lock down the CORS
  `allow_origins=["*"]` in `main.py` to your real frontend domain before
  going live.
- MediaPipe's model files load from Google's CDN at runtime — no need to
  bundle or self-host them, but it does mean the first load needs internet
  access (fine for any normal deployment).

## Next steps / what I'd add next

- **Weekly trend chart in the UI** — the backend endpoint exists
  (`/api/analytics/weekly/{session_id}`), just needs a `recharts` line chart
  component on a `/dashboard` route.
- **AI coaching layer** (per your outline) — a single API call. Take the
  `issue` + recent alert history from `/api/analytics/daily` and pass it to
  Claude or OpenAI with a prompt like "explain why this matters and suggest
  one concrete desk/monitor adjustment." Swap the static `ISSUE_COPY` tips in
  `lib/posture.js` for a generated one, falling back to the static copy if
  the API call fails.
- **Pattern discovery** — once you have a few days of `posture_events` in
  SQLite, a scheduled job (or just a button) that buckets alerts by hour of
  day would directly answer "when do I slip the most."
- **Auth / multi-user** — right now `session_id` is a UUID stuck in
  `localStorage`, which is enough for a single-user deploy but won't survive
  a cleared browser or a second device. Real accounts would need actual auth
  before this scales past "just for me."
- **Recalibration button** — useful if you change chairs/desks mid-day;
  currently you have to Stop and Start again, which works but isn't labeled
  as a recalibration action.
# ErgonomicEye
