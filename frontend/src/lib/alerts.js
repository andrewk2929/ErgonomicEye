// Alert delivery.
//
// The original app's bug was relying on a bundled alert.mp3 resolved
// relative to the wrong working directory. We sidestep that entirely by
// synthesizing the beep with the Web Audio API - zero asset files, zero
// path issues, works the same on any deploy target.

let audioCtx = null;

function getAudioCtx() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  return audioCtx;
}

/** A short, two-tone chime. Deliberately not harsh - this fires often. */
export function playChime() {
  try {
    const ctx = getAudioCtx();
    const now = ctx.currentTime;

    [660, 880].forEach((freq, i) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.value = freq;
      const start = now + i * 0.14;
      gain.gain.setValueAtTime(0, start);
      gain.gain.linearRampToValueAtTime(0.18, start + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.001, start + 0.32);
      osc.connect(gain).connect(ctx.destination);
      osc.start(start);
      osc.stop(start + 0.34);
    });
  } catch (e) {
    console.warn("Could not play alert chime:", e);
  }
}

let notifPermissionRequested = false;

export async function ensureNotificationPermission() {
  if (!("Notification" in window)) return false;
  if (Notification.permission === "granted") return true;
  if (Notification.permission === "denied") return false;
  if (notifPermissionRequested) return false;
  notifPermissionRequested = true;
  const result = await Notification.requestPermission();
  return result === "granted";
}

export function showPopupNotification(title, body) {
  if ("Notification" in window && Notification.permission === "granted") {
    try {
      new Notification(title, { body, icon: "/favicon.svg" });
      return true;
    } catch (e) {
      console.warn("Notification failed:", e);
    }
  }
  return false;
}

/**
 * Fires whichever channels the user has enabled. `kinds` is a Set
 * containing any of "sound" | "popup" | "text". The text channel is
 * handled by the caller (it just renders the banner in the UI) - this
 * function returns whether a text banner should be shown so the caller
 * can decide.
 */
export function fireAlert({ kinds, title, message }) {
  if (kinds.has("sound")) playChime();
  if (kinds.has("popup")) {
    const delivered = showPopupNotification(title, message);
    // Fall back to the in-page banner if the browser notification
    // couldn't be shown (permission denied / unsupported).
    if (!delivered) return true;
  }
  return kinds.has("text");
}
