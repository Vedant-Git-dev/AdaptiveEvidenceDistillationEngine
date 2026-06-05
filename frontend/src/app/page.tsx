"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
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

  // When the run finishes, fetch the raw-gemini baseline for the metrics card.
  useEffect(() => {
    if (run.status !== "done" || !lastPipeline) return;
    const lastUser = [...messages].reverse().find((m) => m.role === "user");
    if (!lastUser) return;
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/api/raw-gemini", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: lastUser.content }),
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
