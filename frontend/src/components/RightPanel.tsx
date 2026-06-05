"use client";

import { useTheme } from "@/components/ThemeProvider";
import { Check, Sun, Moon, AlertTriangle, Loader2 } from "lucide-react";
import type { PipelineState, RawGemini, WorkflowStepUI } from "@/lib/types";
import { aedeTotalTokens, reductionPct } from "@/lib/workflow";
import { Sparkline } from "@/components/Sparkline";
import { cn } from "@/lib/cn";

type Props = {
  steps: WorkflowStepUI[];
  pipeline: PipelineState | null;
  rawGemini: RawGemini | null;
  totalMs: number;
  loading: boolean;
};

export function RightPanel({ steps, pipeline, rawGemini, totalMs, loading }: Props) {
  const { theme, toggle } = useTheme();
  const aede = pipeline ? aedeTotalTokens(pipeline.token_usage) : 0;
  const raw = rawGemini?.tokens ?? 0;
  const reduction = pipeline && rawGemini ? reductionPct(aede, raw) : null;

  return (
    <aside className="flex h-full w-[22rem] shrink-0 flex-col gap-4 overflow-y-auto border-l border-surface-200 bg-surface-50 p-4 dark:border-surface-800 dark:bg-surface-900">
      <header className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold tracking-wide text-surface-500 dark:text-surface-400 uppercase">
            Workflow
          </h2>
          <p className="text-xs text-surface-500">Live view of AEDE's reasoning.</p>
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
        {steps.length === 0 && !loading ? (
          <p className="text-sm text-surface-500">No steps yet — ask a question.</p>
        ) : (
          <ol className="flex flex-col gap-1.5">
            {steps.map((s) => (
              <StepRow key={s.id} step={s} />
            ))}
          </ol>
        )}

        {pipeline && pipeline.coverage_history.length > 0 && (
          <div className="mt-1 text-accent-500">
            <Sparkline values={pipeline.coverage_history} width={296} height={28} />
          </div>
        )}

        {/* Per-step timing table */}
        {steps.some((s) => s.elapsedMs != null) && (
          <div className="mt-4 border-t border-surface-200 pt-3 dark:border-surface-800">
            <h3 className="mb-2 text-[10px] font-semibold tracking-wide text-surface-500 uppercase">
              Timing
            </h3>
            <ul className="flex flex-col gap-1 text-xs">
              {steps
                .filter((s) => s.elapsedMs != null)
                .map((s) => (
                  <li
                    key={s.id}
                    className="flex items-baseline justify-between font-mono"
                  >
                    <span className="text-surface-600 dark:text-surface-400">
                      {s.label}
                    </span>
                    <span className="tabular-nums">{(s.elapsedMs! / 1000).toFixed(2)}s</span>
                  </li>
                ))}
              {totalMs > 0 && (
                <li className="mt-1 flex items-baseline justify-between border-t border-surface-200 pt-1 font-mono font-semibold dark:border-surface-800">
                  <span>Total (AEDE)</span>
                  <span className="tabular-nums">{(totalMs / 1000).toFixed(2)}s</span>
                </li>
              )}
            </ul>
          </div>
        )}
      </section>

      {/* Route chip */}
      {pipeline && (
        <div className="flex items-center gap-2 text-xs text-surface-500">
          <span>Route</span>
          <span
            className={cn(
              "pill",
              pipeline.route === "direct_answer"
                ? "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300"
                : pipeline.route === "max_retrieval_reached"
                  ? "bg-amber-500/15 text-amber-700 dark:text-amber-300"
                  : "bg-accent-500/15 text-accent-700 dark:text-accent-300",
            )}
          >
            {humanRoute(pipeline.route)}
          </span>
        </div>
      )}

      {/* Token Reduction — the hero metric. Card stays blank until BOTH
          AEDE and the raw-Gemini baseline have returned. */}
      <section className="card">
        <h3 className="mb-1 text-xs font-semibold tracking-wide text-surface-500 dark:text-surface-400 uppercase">
          Token Reduction
        </h3>
        {pipeline && rawGemini ? (
          <>
            <div className="grid grid-cols-3 gap-2">
              <Metric
                label="Raw Gemini"
                value={raw > 0 ? raw.toLocaleString() : "—"}
                tone="muted"
              />
              <Metric
                label="AEDE"
                value={aede > 0 ? aede.toLocaleString() : "—"}
                tone="accent"
              />
              <Metric
                label="Reduction"
                value={reduction != null ? `${reduction}%` : "—"}
                tone={reduction != null && reduction > 0 ? "good" : "muted"}
              />
            </div>
            {raw > 0 && aede > 0 && (
              <div className="mt-3">
                <ReductionBar aede={aede} raw={raw} />
              </div>
            )}
          </>
        ) : (
          <div className="flex items-center gap-2 py-3 text-sm text-surface-500">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            <span>
              {loading
                ? "Waiting for AEDE and the raw-Gemini baseline…"
                : "Run a query to see how much AEDE saved vs. raw Gemini."}
            </span>
          </div>
        )}
      </section>

      {pipeline?.error && (
        <div className="flex items-start gap-2 rounded-lg border border-red-300 bg-red-50 p-3 text-xs text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>{pipeline.error}</span>
        </div>
      )}
    </aside>
  );
}

function StepRow({ step }: { step: WorkflowStepUI }) {
  const isRunning = step.status === "running";
  return (
    <li
      className={cn(
        "flex items-center gap-2.5 rounded-md px-1.5 py-1.5 text-sm transition",
        isRunning && "bg-accent-500/5",
      )}
    >
      <span
        className={cn(
          "flex h-5 w-5 shrink-0 items-center justify-center rounded-full",
          isRunning
            ? "bg-accent-500/15 text-accent-600 dark:text-accent-400"
            : "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400",
        )}
      >
        {isRunning ? (
          <Loader2 className="h-3 w-3 animate-spin" />
        ) : (
          <Check className="h-3 w-3" strokeWidth={3} />
        )}
      </span>
      <div className="flex min-w-0 flex-1 items-center gap-2">
        <span className="truncate font-medium">{step.label}</span>
        <span className="truncate text-[10px] text-surface-500">
          ({step.model})
        </span>
      </div>
      {step.detail && (
        <span className="font-mono text-[10px] text-surface-500">
          {step.detail}
        </span>
      )}
      {step.elapsedMs != null && (
        <span className="font-mono text-[10px] tabular-nums text-surface-500">
          {(step.elapsedMs / 1000).toFixed(2)}s
        </span>
      )}
    </li>
  );
}

function ReductionBar({ aede, raw }: { aede: number; raw: number }) {
  const aedePct = Math.max(2, Math.min(100, (aede / Math.max(raw, 1)) * 100));
  const savedPct = 100 - aedePct;
  return (
    <div>
      <div className="flex h-2 w-full overflow-hidden rounded-full bg-surface-200 dark:bg-surface-800">
        <div
          className="h-full bg-accent-500 transition-all"
          style={{ width: `${aedePct}%` }}
        />
      </div>
      <div className="mt-1 flex justify-between text-[10px] text-surface-500">
        <span>AEDE {aedePct.toFixed(0)}%</span>
        <span className="text-emerald-600 dark:text-emerald-400">
          Saved {savedPct.toFixed(0)}%
        </span>
      </div>
    </div>
  );
}

function humanRoute(route: string) {
  switch (route) {
    case "direct_answer":
      return "Direct Answer";
    case "llama_with_compress":
      return "Compressed + Llama";
    case "deep_reasoning":
      return "Deep Reasoning";
    case "answer":
      return "Compress → Gemini";
    case "retrieve_more":
      return "Retrieve More";
    case "max_retrieval_reached":
      return "Max Retrieval";
    default:
      return route || "—";
  }
}

function Metric({
  label,
  value,
  tone,
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
      <span className="text-[10px] font-medium tracking-wide text-surface-500 uppercase">
        {label}
      </span>
      <span className={cn("stat-num", toneClass)}>{value}</span>
    </div>
  );
}
