"use client";

import { useEffect, useRef, useState } from "react";
import { FileText, Upload, Loader2 } from "lucide-react";
import { cn } from "@/lib/cn";
import type { StatsResponse } from "@/lib/types";

type Props = {
  stats: StatsResponse | null;
  onStats: (s: StatsResponse) => void;
};

export function LeftPanel({ stats, onStats }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filename, setFilename] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/stats")
      .then((r) => (r.ok ? r.json() : null))
      .then((d: StatsResponse | null) => d && onStats(d))
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const upload = async (file: File) => {
    setError(null);
    setUploading(true);
    setFilename(file.name);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch("/api/upload", { method: "POST", body: fd });
      if (!res.ok) {
        const detail = (await res.json().catch(() => ({}))).detail ?? res.statusText;
        throw new Error(detail);
      }
      const data = (await res.json()) as StatsResponse;
      onStats(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  return (
    <aside className="flex h-full w-80 shrink-0 flex-col gap-4 border-r border-surface-200 bg-surface-50 p-4 dark:border-surface-800 dark:bg-surface-900">
      <header className="flex items-center gap-2">
        <AedeMark />
        <div>
          <h1 className="text-sm font-semibold tracking-wide text-surface-900 dark:text-surface-100">
            AEDE
          </h1>
          <p className="text-[10px] uppercase tracking-wider text-surface-500">
            Adaptive Evidence-Driven Extraction
          </p>
        </div>
      </header>

      {/* Upload zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const f = e.dataTransfer.files?.[0];
          if (f) void upload(f);
        }}
        onClick={() => inputRef.current?.click()}
        className={cn(
          "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed p-6 text-center transition",
          dragging
            ? "border-accent-500 bg-accent-500/5"
            : "border-surface-300 hover:border-accent-400 dark:border-surface-700",
        )}
      >
        {uploading ? (
          <Loader2 className="h-6 w-6 animate-spin text-accent-500" />
        ) : (
          <Upload className="h-6 w-6 text-surface-400" />
        )}
        <p className="text-sm font-medium">
          {uploading ? "Ingesting…" : "Drop a PDF or click to upload"}
        </p>
        {filename && !uploading && (
          <p className="flex items-center gap-1 text-xs text-surface-500">
            <FileText className="h-3 w-3" /> {filename}
          </p>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void upload(f);
          }}
        />
      </div>
      {error && <p className="text-xs text-red-500">{error}</p>}

      {/* Document Stats */}
      <section className="card flex-1">
        <h2 className="mb-3 text-xs font-semibold tracking-wide text-surface-500 uppercase">
          Document Stats
        </h2>
        {stats ? (
          <dl className="grid grid-cols-1 gap-3">
            <Stat label="Pages" value={stats.pages} />
            <Stat label="Chunks" value={stats.chunks} />
            <Stat label="Embeddings" value={stats.embeddings} />
          </dl>
        ) : (
          <p className="text-sm text-surface-500">
            Loading…
          </p>
        )}
      </section>
    </aside>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-baseline justify-between">
      <dt className="text-sm text-surface-500 dark:text-surface-400">{label}</dt>
      <dd className="stat-num">{value.toLocaleString()}</dd>
    </div>
  );
}

function AedeMark() {
  return (
    <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-accent-500 to-emerald-500 text-white shadow-sm">
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="h-4 w-4"
      >
        <path d="M4 19V5l8 7 8-7v14" />
      </svg>
    </div>
  );
}
