# ErgonomicEye 2.0

Real-time posture and sedentary-behavior monitoring that actually works as a
deployed website. 

# Deployed at [https://ergonomic-eye.vercel.app/](url)

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

# ErgonomicEye
