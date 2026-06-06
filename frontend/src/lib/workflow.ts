export type WorkflowStep = {
  id: string;
  label: string;
  detail?: string;
  /** Short model label, e.g. "Llama 3.1 8B" or "Gemini 2.5 Flash". */
  model?: string;
};

export type StepTiming = {
  node: string;
  model: string;
  elapsed_ms: number;
};

/**
 * Canonical model label for each pipeline step. The backend's STEP_MODELS
 * is the source of truth at runtime; this client-side map gives a stable
 * label for steps that don't have a corresponding timing entry (e.g. the
 * retrieve step emits timings under node=retrieve, not focused_retriever).
 */
const NODE_MODELS: Record<string, string> = {
  extract_concepts: "Keyword extraction",
  focused_retriever: "ChromaDB",
  retrieve_more: "ChromaDB",
  extract: "Llama",
  analyze: "Llama",
  compress: "Llama",
  compile: "Compiler",
  reason: "Gemini",
  small_reasoner: "Llama",
};

/** Parse a backend workflow_path into display rows. */
export function parseWorkflow(
  path: string[],
  _timings: StepTiming[] = [],
): WorkflowStep[] {
  const out: WorkflowStep[] = [];
  const counts: Record<string, number> = {};
  const bump = (id: string) => {
    counts[id] = (counts[id] ?? 0) + 1;
    return counts[id] > 1 ? `${id}-${counts[id]}` : id;
  };

  for (const raw of path) {
    if (raw === "start") continue;
    if (raw === "focused_retriever") {
      out.push({ id: bump("retrieve"), label: "Retrieve", detail: "k=4", model: NODE_MODELS.focused_retriever });
      continue;
    }
    if (raw.startsWith("retrieve_more")) {
      const m = raw.match(/\(k=(\d+)\)/);
      out.push({
        id: bump("retrieve-more"),
        label: "Retrieve more",
        detail: m ? `k=${m[1]}` : undefined,
        model: NODE_MODELS.retrieve_more,
      });
      continue;
    }
    if (raw === "extract") {
      out.push({ id: bump("extract"), label: "Extract", model: NODE_MODELS.extract });
      continue;
    }
    if (raw === "analyze") {
      out.push({ id: bump("analyze"), label: "Analyze", model: NODE_MODELS.analyze });
      continue;
    }
    if (raw === "compress") {
      out.push({ id: bump("compress"), label: "Compress", model: NODE_MODELS.compress });
      continue;
    }
    if (raw.startsWith("compile(")) {
      const decision = raw.slice("compile(".length, -1);
      out.push({
        id: bump("compile"),
        label: "Compile",
        detail: decision,
        model: NODE_MODELS.compile,
      });
      continue;
    }
    if (raw === "reason" || raw === "final_reasoner") {
      out.push({ id: bump("reason"), label: "Reason", model: NODE_MODELS.reason });
      continue;
    }
    if (raw === "small_reasoner") {
      out.push({ id: bump("small-reasoner"), label: "Direct Answer", model: NODE_MODELS.small_reasoner });
      continue;
    }
    out.push({ id: bump(raw), label: raw });
  }
  return out;
}
