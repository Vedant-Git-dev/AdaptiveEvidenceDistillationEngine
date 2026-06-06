"use client";

import { Sun, Moon, AlertTriangle, Loader2 } from "lucide-react";
import { useTheme } from "@/components/ThemeProvider";
import { parseWorkflow } from "@/lib/workflow";
import { cn } from "@/lib/cn";
import type { OptimizeResponse } from "@/lib/types";

type Props = {
  result: OptimizeResponse | null;
  loading: boolean;
  error: string | null;
};

export function RightPanel({ result, loading, error }: Props) {
  const { theme, toggle } = useTheme();

  return (
    <aside className="flex h-full w-[22rem] shrink-0 flex-col gap-4 overflow-y-auto border-l border-surface-200 bg-surface-50 p-4 dark:border-surface-800 dark:bg-surface-900">
      <header className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold tracking-wide text-surface-500 dark:text-surface-400 uppercase">
            Workflow
          </h2>
          <p className="text-xs text-surface-500">AEDE's last run.</p>
        </div>
        <button
          onClick={toggle}
          aria-label="Toggle theme"
          className="rounded-md border border-surface-300 p-1.5 text-surface-500 hover:text-surface-900 dark:border-surface-700 dark:hover:text-white"
        >
          {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </button>
      </header>

      {/* Steps */}
      <section className="card">
        {result ? (
          <ol className="flex flex-col gap-1.5">
            {parseWorkflow(result.workflow_path, result.timings).map((s) => (
              <li
                key={s.id}
                className="flex items-center gap-2 rounded-md px-1.5 py-1 text-sm"
              >
                <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-emerald-500/15 text-emerald-600 dark:text-emerald-400">
                  ✓
                </span>
                <span className="font-medium">{s.label}</span>
                {s.model && (
                  <span className="truncate text-[10px] text-surface-500">
                    ({s.model})
                  </span>
                )}
                {s.detail && (
                  <span className="ml-auto font-mono text-[10px] text-surface-500">{s.detail}</span>
                )}
              </li>
            ))}
          </ol>
        ) : loading ? (
          <div className="flex items-center gap-2 py-3 text-sm text-surface-500">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            <span>Running AEDE pipeline…</span>
          </div>
        ) : (
          <p className="text-sm text-surface-500">No steps yet — ask a question.</p>
        )}

        {result && (
          <div className="mt-3 flex items-center gap-2 text-xs text-surface-500">
            <span>Route</span>
            <span className={cn("pill", routeTone(result.decision))}>
              {humanRoute(result.decision)}
            </span>
          </div>
        )}
      </section>

      {/* Per-step timing */}
      {result && result.timings && result.timings.length > 0 && (
        <section className="card">
          <h3 className="mb-2 text-xs font-semibold tracking-wide text-surface-500 dark:text-surface-400 uppercase">
            Timing
          </h3>
          <ul className="flex flex-col gap-1 text-xs">
            {result.timings.map((t, i) => (
              <li
                key={`${t.node}-${i}`}
                className="flex items-baseline justify-between font-mono"
              >
                <span className="text-surface-600 dark:text-surface-400">
                  {labelForNode(t.node)}
                </span>
                <span className="tabular-nums">{(t.elapsed_ms / 1000).toFixed(2)}s</span>
              </li>
            ))}
            {result.total_ms > 0 && (
              <li className="mt-1 flex items-baseline justify-between border-t border-surface-200 pt-1 font-mono font-semibold dark:border-surface-800">
                <span>Total (AEDE)</span>
                <span className="tabular-nums">{(result.total_ms / 1000).toFixed(2)}s</span>
              </li>
            )}
          </ul>
        </section>
      )}

      {/* Token Reduction */}
      <section className="card">
        <h3 className="mb-1 text-xs font-semibold tracking-wide text-surface-500 dark:text-surface-400 uppercase">
          Token Reduction
        </h3>
        {result ? (
          <>
            <div className="grid grid-cols-3 gap-2">
              <Metric
                label="Final model"
                value={result.final_tokens > 0 ? result.final_tokens.toLocaleString() : "—"}
                tone="accent"
              />
              <Metric
                label="Raw Gemini"
                value={result.raw_tokens > 0 ? result.raw_tokens.toLocaleString() : "—"}
                tone="muted"
              />
              <Metric
                label="Saved"
                value={result.raw_tokens > 0 ? `${Math.round(result.saved_pct * 100)}%` : "—"}
                tone={result.saved_pct > 0 ? "good" : "muted"}
              />
            </div>
            {result.raw_tokens > 0 && result.final_tokens > 0 && (
              <div className="mt-3">
                <ReductionBar aede={result.final_tokens} raw={result.raw_tokens} />
              </div>
            )}
            <div className="mt-3 text-[10px] text-surface-500">
              Items: {result.items_count} · Coverage: {(result.coverage * 100).toFixed(0)}%
            </div>
          </>
        ) : loading ? (
          <div className="flex items-center gap-2 py-3 text-sm text-surface-500">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            <span>Waiting for results…</span>
          </div>
        ) : (
          <p className="text-sm text-surface-500">Run a query to see how much AEDE saved vs. raw Gemini.</p>
        )}
      </section>

      {error && (
        <div className="flex items-start gap-2 rounded-lg border border-red-300 bg-red-50 p-3 text-xs text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </aside>
  );
}

function routeTone(decision: string): string {
  if (decision === "direct_answer") return "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300";
  if (decision === "max_retrieval_reached") return "bg-amber-500/15 text-amber-700 dark:text-amber-300";
  return "bg-accent-500/15 text-accent-700 dark:text-accent-300";
}

function humanRoute(decision: string): string {
  switch (decision) {
    case "direct_answer":
      return "Direct Answer";
    case "llama_with_compress":
      return "Compressed + Llama";
    case "deep_reasoning":
      return "Deep Reasoning";
    case "answer":
    case "compress":
      return "Compress → Gemini";
    case "retrieve_more":
      return "Retrieve More";
    case "max_retrieval_reached":
      return "Max Retrieval";
    default:
      return decision || "—";
  }
}

function ReductionBar({ aede, raw }: { aede: number; raw: number }) {
  const aedePct = Math.max(2, Math.min(100, (aede / Math.max(raw, 1)) * 100));
  const savedPct = 100 - aedePct;
  return (
    <div>
      <div className="flex h-2 w-full overflow-hidden rounded-full bg-surface-200 dark:bg-surface-800">
        <div className="h-full bg-accent-500 transition-all" style={{ width: `${aedePct}%` }} />
      </div>
      <div className="mt-1 flex justify-between text-[10px] text-surface-500">
        <span>AEDE {aedePct.toFixed(0)}%</span>
        <span className="text-emerald-600 dark:text-emerald-400">Saved {savedPct.toFixed(0)}%</span>
      </div>
    </div>
  );
}

const NODE_LABELS: Record<string, string> = {
  extract_concepts: "Extract concepts",
  focused_retriever: "Retrieve",
  retrieve: "Retrieve",
  retrieve_more: "Retrieve more",
  extract: "Extract",
  analyze: "Analyze",
  compile: "Compile",
  compress: "Compress",
  reason: "Reason (Gemini)",
  small_reasoner: "Direct Answer",
};

function labelForNode(node: string): string {
  return NODE_LABELS[node] ?? node;
}

function Metric({
  label, value, tone,
}: {
  label: string;
  value: string;
  tone: "muted" | "accent" | "good";
}) {
  const toneClass =
    tone === "good"
      ? "text-emerald-600 dark:text-emerald-400"
      : tone === "accent"
        ? "text-accent-600 dark:text-accent-400"
        : "text-surface-700 dark:text-surface-300";
  return (
    <div className="flex flex-col">
      <span className="text-[10px] font-medium tracking-wide text-surface-500 uppercase">{label}</span>
      <span className={cn("stat-num", toneClass)}>{value}</span>
    </div>
  );
}
