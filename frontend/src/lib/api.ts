// Single source of truth for the backend URL. Direct to :8000 so we
// skip the Next.js dev proxy (which has a hard 30s timeout).
// Note: the backend mounts endpoints at the root (e.g. /collections),
// not under /api. The Next.js dev proxy stripped /api before forwarding.
export const API_BASE = "http://127.0.0.1:8000";
