"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";
import type {
  ChatMessage,
  PipelineState,
  RawGemini,
  WorkflowStepUI,
} from "@/lib/types";

let msgCounter = 0;
const newId = () => `m_${Date.now()}_${++msgCounter}`;

type Status = "idle" | "running" | "done" | "error";

type RunState = {
  status: Status;
  steps: WorkflowStepUI[];
  tokenUsage: Record<string, number>;
  coverageHistory: number[];
  coverage: number;
  currentTopK: number;
  totalMs: number;
  errorMessage?: string;
};

const emptyRun: RunState = {
  status: "idle",
  steps: [],
  tokenUsage: {},
  coverageHistory: [],
  coverage: 0,
  currentTopK: 0,
  totalMs: 0,
};

/** Step labels — keep in sync with backend `STEP_MODELS` node keys. */
const NODE_LABEL: Record<string, string> = {
  extract_concepts: "Extract concepts",
  focused_retriever: "Retrieve",
  retrieve: "Retrieve",
  retrieve_more: "Retrieve more",
  extract: "Extract",
  analyze: "Analyze",
  compress: "Compress",
  compile: "Compile",
  reason: "Reason",
  small_reasoner: "Direct Answer",
};

function parseCompileDecision(entry: string): string | undefined {
  const m = entry.match(/^compile\(([^)]+)\)$/);
  return m?.[1];
}

export function useChatStream(opts: {
  onAssistantMessage: (m: ChatMessage) => void;
}) {
  const { onAssistantMessage } = opts;
  const [run, setRun] = useState<RunState>(emptyRun);
  const [isRunning, setIsRunning] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  // Use a ref for the callback so the SSE loop never sees a stale closure,
  // and so we don't need to put onAssistantMessage in any useCallback dep.
  const callbackRef = useRef(opts.onAssistantMessage);
  useEffect(() => {
    callbackRef.current = opts.onAssistantMessage;
  }, [opts.onAssistantMessage]);

  const send = useCallback(async (query: string) => {
    if (!query.trim() || isRunning) return;

    // Append the user message via the ref — never via the captured closure.
    callbackRef.current({ id: newId(), role: "user", content: query });

    // Synchronously transition UI into "running" so the spinner shows immediately.
    flushSync(() => {
      setIsRunning(true);
      setRun({ ...emptyRun, status: "running" });
    });

    const ctrl = new AbortController();
    abortRef.current = ctrl;

    let finalAnswer: string | null = null;
    let finalPipeline: PipelineState | null = null;
    let streamError: string | null = null;

    try {
      const res = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) {
        throw new Error(`Stream failed: ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let sepIdx;
        while ((sepIdx = buffer.indexOf("\n\n")) !== -1) {
          const frame = buffer.slice(0, sepIdx);
          buffer = buffer.slice(sepIdx + 2);
          const ev = parseFrame(frame);
          if (!ev) continue;
          handleEvent(ev);
          // Yield to the browser so it can paint between rapid events.
          // Without this, flushSync can starve the event loop and the user
          // never sees the intermediate state.
          await new Promise<void>((r) => setTimeout(r, 0));
        }
      }
    } catch (e) {
      streamError = e instanceof Error ? e.message : String(e);
    } finally {
      setIsRunning(false);
      abortRef.current = null;
    }

    // Now that the stream is done, transition UI and append the assistant
    // message exactly once. Doing this AFTER the stream closes (instead of
    // inside a setRun updater) avoids React strict-mode double-invocation
    // duplicating the message.
    if (streamError) {
      flushSync(() => {
        setRun((prev) => ({
          ...prev,
          status: "error",
          errorMessage: streamError ?? undefined,
        }));
      });
    }
    // The 'done' SSE handler above already flushed the final run state; no
    // second flush needed here.

    if (streamError) {
      callbackRef.current({
        id: newId(),
        role: "assistant",
        content: `Error: ${streamError}`,
      });
    } else if (finalAnswer !== null && finalPipeline) {
      callbackRef.current({
        id: newId(),
        role: "assistant",
        content: finalAnswer,
        pipeline: finalPipeline,
      });
    }

    function handleEvent(ev: { event: string; data: any }) {
      if (ev.event === "started") return;

      if (ev.event === "step_started") {
        const d = ev.data as { node: string; model: string };
        flushSync(() => {
          setRun((prev) => {
            const steps = prev.steps.filter((s) => s.status !== "running");
            steps.push({
              id: `${d.node}_${steps.length + 1}_${Date.now()}`,
              node: d.node,
              label: NODE_LABEL[d.node] ?? d.node,
              model: d.model || "—",
              status: "running",
            });
            return { ...prev, steps };
          });
        });
        return;
      }

      if (ev.event === "step_finished") {
        const d = ev.data as {
          node: string;
          model: string;
          elapsed_ms: number;
          workflow_path: string[];
        };
        flushSync(() => {
          setRun((prev) => {
            const steps = prev.steps.slice();
            const idx = (() => {
              for (let i = steps.length - 1; i >= 0; i--) {
                if (steps[i].node === d.node && steps[i].status === "running") {
                  return i;
                }
              }
              return -1;
            })();
            if (idx >= 0) {
              const cur = steps[idx];
              const updated: WorkflowStepUI = {
                ...cur,
                status: "done",
                elapsedMs: d.elapsed_ms,
                model: d.model || cur.model,
              };
              if (d.node === "compile" && d.workflow_path?.length) {
                const tail = d.workflow_path[d.workflow_path.length - 1];
                const dec = parseCompileDecision(tail);
                if (dec) {
                  updated.decision = dec;
                  updated.detail = dec;
                }
              } else if (d.node === "retrieve_more" && d.workflow_path?.length) {
                const tail = d.workflow_path[d.workflow_path.length - 1];
                const m = tail.match(/\(k=(\d+)\)/);
                if (m) updated.detail = `k=${m[1]}`;
              }
              steps[idx] = updated;
            }
            return { ...prev, steps };
          });
        });
        return;
      }

      if (ev.event === "done") {
        const d = ev.data as { state: PipelineState; total_ms: number };
        finalAnswer = d.state.answer || "(no answer)";
        finalPipeline = d.state;
        // Finalize totalMs into run state immediately so the right panel
        // shows the Total as soon as the run closes.
        flushSync(() => {
          setRun((prev) => ({
            ...prev,
            status: "done",
            totalMs: d.total_ms,
            tokenUsage: d.state.token_usage,
            coverage: d.state.coverage,
            coverageHistory: d.state.coverage_history,
            currentTopK: d.state.current_top_k,
          }));
        });
        return;
      }

      if (ev.event === "error") {
        const d = ev.data as { message: string };
        streamError = d.message;
        return;
      }
    }
  }, [isRunning]);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return { run, isRunning, send, cancel };
}

// --- helpers ---

function parseFrame(frame: string): { event: string; data: any } | null {
  const lines = frame.split("\n");
  let event = "message";
  let dataStr = "";
  for (const line of lines) {
    if (line.startsWith("event: ")) event = line.slice(7).trim();
    else if (line.startsWith("data: ")) dataStr += line.slice(6);
  }
  if (!dataStr) return null;
  try {
    return { event, data: JSON.parse(dataStr) };
  } catch {
    return { event, data: dataStr };
  }
}

// Silence "unused" lint for the old helper. (kept exported for compat)
export { parseCompileDecision };
export type { RawGemini };
