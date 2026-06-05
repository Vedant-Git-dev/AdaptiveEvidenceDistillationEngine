"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { LeftPanel } from "@/components/LeftPanel";
import { ChatPanel } from "@/components/ChatPanel";
import { RightPanel } from "@/components/RightPanel";
import { useChatStream } from "@/lib/useChatStream";
import type {
  ChatMessage,
  PipelineState,
  RawGemini,
  StatsResponse,
} from "@/lib/types";

export default function Page() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [lastPipeline, setLastPipeline] = useState<PipelineState | null>(null);
  const [lastRaw, setLastRaw] = useState<RawGemini | null>(null);

  const handleMessage = useCallback((m: ChatMessage) => {
    setMessages((prev) => [...prev, m]);
  }, []);

  const { run, isRunning, send } = useChatStream({
    onAssistantMessage: (m) => {
      handleMessage(m);
      if (m.pipeline) setLastPipeline(m.pipeline);
    },
  });

  // Track the last user query in a ref so the raw-gemini effect can read
  // it without depending on the `messages` array (which is in flux during
  // the run). This is the only way to be sure the effect sees the query
  // the moment `run.status` flips to "done".
  const lastQueryRef = useRef<string | null>(null);
  useEffect(() => {
    // Update the ref whenever a new user message is appended.
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "user") {
        lastQueryRef.current = messages[i].content;
        break;
      }
    }
  }, [messages]);

  // When the run finishes, fetch the raw-gemini baseline for the metrics card.
  useEffect(() => {
    if (run.status !== "done" || !lastPipeline) return;
    const query = lastQueryRef.current;
    if (!query) return;
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/api/raw-gemini", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query }),
        });
        if (res.ok && !cancelled) {
          const r = (await res.json()) as RawGemini;
          setLastRaw(r);
        }
      } catch {
        /* ignore */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [run.status, lastPipeline]);

  const currentStep = useMemo(() => {
    for (let i = run.steps.length - 1; i >= 0; i--) {
      if (run.steps[i].status === "running") return run.steps[i];
    }
    return null;
  }, [run.steps]);

  const onStats = useCallback((s: StatsResponse) => setStats(s), []);

  return (
    <div className="flex h-screen w-screen overflow-hidden">
      <LeftPanel stats={stats} onStats={onStats} />
      <ChatPanel
        messages={messages}
        onSend={send}
        isRunning={isRunning}
        currentStep={currentStep}
        hasDocument={(stats?.embeddings ?? 0) > 0}
      />
      <RightPanel
        steps={run.steps}
        pipeline={lastPipeline}
        rawGemini={lastRaw}
        totalMs={run.totalMs}
        loading={isRunning}
      />
    </div>
  );
}
