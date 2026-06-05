import type { PipelineState } from "@/lib/types";

export type WorkflowStep = {
  /** Stable id used as React key. */
  id: string;
  /** Short label shown to the user. */
  label: string;
  /** Optional detail (e.g. "k=4"). */
  detail?: string;
};

/**
 * Parse the AEDE workflow_path into a normalized, deduplicated list of steps
 * suitable for rendering. The graph emits raw node names; we collapse repeated
 * visits (e.g. extract/analyze on the retrieve-more loop) into a single "ran"
 * step with a count, so the panel stays compact.
 *
 * Example path:
 *   ["start","focused_retriever","extract","analyze","compile(retrieve_more)",
 *    "retrieve_more(k=8)","extract","analyze","compile(direct_answer)","small_reasoner"]
 *
 * Becomes:
 *   [
 *     { id: "retrieve", label: "Retrieve", detail: "k=4" },
 *     { id: "extract",  label: "Extract" },
 *     { id: "analyze",  label: "Analyze" },
 *     { id: "compiler-1", label: "Compile", detail: "retrieve_more" },
 *     { id: "retrieve-more-1", label: "Retrieve more", detail: "k=8" },
 *     { id: "extract-2", label: "Extract" },
 *     { id: "analyze-2", label: "Analyze" },
 *     { id: "compiler-2", label: "Compile", detail: "direct_answer" },
 *     { id: "answer", label: "Direct Answer" },
 *   ]
 */
export function parseWorkflow(path: string[]): WorkflowStep[] {
  const out: WorkflowStep[] = [];
  const counts: Record<string, number> = {};

  const bump = (id: string) => {
    counts[id] = (counts[id] ?? 0) + 1;
    return counts[id] > 1 ? `${id}-${counts[id]}` : id;
  };

  for (const raw of path) {
    if (raw === "start") continue;

    if (raw === "focused_retriever") {
      out.push({ id: bump("retrieve"), label: "Retrieve", detail: "k=4" });
      continue;
    }
    if (raw.startsWith("retrieve_more")) {
      const m = raw.match(/\(k=(\d+)\)/);
      out.push({
        id: bump("retrieve-more"),
        label: "Retrieve more",
        detail: m ? `k=${m[1]}` : undefined,
      });
      continue;
    }
    if (raw === "extract") {
      out.push({ id: bump("extract"), label: "Extract" });
      continue;
    }
    if (raw === "analyze") {
      out.push({ id: bump("analyze"), label: "Analyze" });
      continue;
    }
    if (raw === "compress") {
      out.push({ id: bump("compress"), label: "Compress" });
      continue;
    }
    if (raw.startsWith("compile(")) {
      const decision = raw.slice("compile(".length, -1);
      out.push({ id: bump("compile"), label: "Compile", detail: decision });
      continue;
    }
    if (raw === "reason" || raw === "final_reasoner") {
      out.push({ id: bump("reason"), label: "Reason (Gemini)" });
      continue;
    }
    if (raw === "small_reasoner") {
      out.push({ id: bump("small-reasoner"), label: "Direct Answer" });
      continue;
    }
    // Fallback: show the raw name.
    out.push({ id: bump(raw), label: raw });
  }

  return out;
}

export function aedeTotalTokens(usage: PipelineState["token_usage"]): number {
  // Prefer a pre-computed total if the backend provides one.
  if (typeof usage.total === "number" && usage.total > 0) return usage.total;
  // Sum every numeric field except the obvious subtotals we don't want to
  // double-count (aede backend already rolls up "total" in many runs).
  let sum = 0;
  for (const [k, v] of Object.entries(usage)) {
    if (k === "total" || typeof v !== "number") continue;
    sum += v;
  }
  return sum;
}

export function reductionPct(aede: number, raw: number): number {
  if (!raw) return 0;
  return Math.max(0, Math.min(100, Math.round((1 - aede / raw) * 100)));
}
