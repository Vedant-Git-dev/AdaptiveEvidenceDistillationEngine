"use client";

import { useCallback, useRef, useState } from "react";
import type { OptimizeResponse } from "@/lib/types";
import { API_BASE } from "@/lib/api";

type Status = "idle" | "running" | "done" | "error";

type RunState = {
  status: Status;
  result: OptimizeResponse | null;
  errorMessage: string | null;
};

const empty: RunState = { status: "idle", result: null, errorMessage: null };

export function useOptimize() {
  const [run, setRun] = useState<RunState>(empty);
  const abortRef = useRef<AbortController | null>(null);

  const send = useCallback(async (collections: string[], query: string) => {
    if (!query.trim() || collections.length === 0) return;
    setRun({ status: "running", result: null, errorMessage: null });
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      const res = await fetch(`${API_BASE}/optimize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ collections, query }),
        signal: ctrl.signal,
      });
      if (!res.ok) {
        let detail = res.statusText;
        try {
          const body = await res.json();
          detail = body.detail ?? JSON.stringify(body);
        } catch {
          // body might be HTML (proxy 500) or empty
          try {
            const text = await res.text();
            if (text) detail = text.slice(0, 200);
          } catch { /* ignore */ }
        }
        throw new Error(`HTTP ${res.status}: ${detail}`);
      }
      const result = (await res.json()) as OptimizeResponse;
      setRun({ status: "done", result, errorMessage: null });
    } catch (e) {
      if ((e as Error).name === "AbortError") {
        setRun(empty);
        return;
      }
      const msg = (e as Error).message || String(e);
      setRun({ status: "error", result: null, errorMessage: msg });
    } finally {
      abortRef.current = null;
    }
  }, []);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return { run, isRunning: run.status === "running", send, cancel };
}

// ----- collections API (fetch list, add, delete) -----

async function extractError(res: Response): Promise<string> {
  // Try JSON first, then plain text, then statusText. The Next.js dev proxy
  // returns 500 with an HTML body when the upstream is down or times out;
  // a useful error string beats a generic "Internal Server Error".
  try {
    const body = await res.json();
    return body.detail ?? JSON.stringify(body);
  } catch {
    try {
      const text = await res.text();
      if (text) return text.slice(0, 200);
    } catch { /* ignore */ }
    return res.statusText || `HTTP ${res.status}`;
  }
}

export async function fetchCollections(): Promise<
  { name: string; display_name: string; type: "pdf" | "paste"; kind?: "chat" | "agent"; items: number }[]
> {
  const res = await fetch(`${API_BASE}/collections`);
  if (!res.ok) {
    throw new Error(`Could not load collections: ${await extractError(res)}`);
  }
  return res.json();
}

export async function addCollection(
  kind: "pdf" | "chat" | "agent",
  payload: { name: string; file?: File; text?: string },
): Promise<unknown> {
  const fd = new FormData();
  let url: string;
  if (kind === "pdf") {
    fd.append("name", payload.name);
    if (payload.file) fd.append("file", payload.file);
    url = `${API_BASE}/collections/pdf`;
  } else {
    fd.append("name", payload.name);
    fd.append("text", payload.text ?? "");
    fd.append("kind", kind);
    url = `${API_BASE}/collections/paste`;
  }
  const res = await fetch(url, { method: "POST", body: fd });
  if (!res.ok) {
    throw new Error(await extractError(res));
  }
  return res.json();
}

export async function deleteCollection(name: string): Promise<void> {
  const res = await fetch(`${API_BASE}/collections/${encodeURIComponent(name)}`, { method: "DELETE" });
  if (!res.ok) {
    throw new Error(await extractError(res));
  }
}
