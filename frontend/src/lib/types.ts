export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  pipeline?: PipelineState;
};

export type PipelineState = {
  answer: string;
  workflow_path: string[];
  route: string;
  coverage: number;
  coverage_history: number[];
  token_usage: Record<string, number>;
  current_top_k: number;
  required_reasoning: "none" | "light" | "deep";
  max_retrieval_reached: boolean;
  error: string | null;
};

export type RawGemini = {
  tokens: number;
  model: string;
  retrieved_docs: number;
};

export type StatsResponse = {
  pages: number;
  chunks: number;
  embeddings: number;
  last_filename?: string | null;
};

/** A step in the right-panel workflow, augmented with model + timing. */
export type WorkflowStepUI = {
  id: string;
  /** Canonical node key (focused_retriever, extract, etc.) */
  node: string;
  label: string;
  model: string;
  /** ms; undefined while in flight. */
  elapsedMs?: number;
  status: "running" | "done";
  detail?: string;
  /** When the node is `compile`, this is the decision. */
  decision?: string;
};
