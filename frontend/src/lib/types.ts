export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  result?: OptimizeResponse;
};

/** Response from POST /optimize. */
export type OptimizeResponse = {
  answer: string;
  decision: string;
  final_tokens: number;
  raw_tokens: number;
  saved_pct: number;
  items_count: number;
  collections_used: string[];
  trace: string[];
  coverage: number;
  workflow_path: string[];
  timings: { node: string; model: string; elapsed_ms: number }[];
  total_ms: number;
};
