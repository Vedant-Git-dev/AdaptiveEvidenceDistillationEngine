"use client";

import { useCallback, useEffect, useState } from "react";
import { LeftPanel, type AddKind, type CollectionItem } from "@/components/LeftPanel";
import { ChatPanel } from "@/components/ChatPanel";
import { RightPanel } from "@/components/RightPanel";
import { addCollection, deleteCollection, fetchCollections, useOptimize } from "@/lib/useOptimize";
import type { ChatMessage } from "@/lib/types";

let msgCounter = 0;
const newId = () => `m_${Date.now()}_${++msgCounter}`;

export default function Page() {
  const [collections, setCollections] = useState<CollectionItem[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [addBusy, setAddBusy] = useState(false);
  const [addError, setAddError] = useState<string | null>(null);

  const { run, isRunning, send } = useOptimize();

  const refresh = useCallback(async () => {
    const data = await fetchCollections();
    setCollections(data as CollectionItem[]);
    // Drop selections that no longer exist
    setSelected((prev) => {
      const names = new Set(data.map((c) => c.name));
      return new Set([...prev].filter((n) => names.has(n)));
    });
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const onAdd = useCallback(
    async (kind: AddKind, payload: { name: string; file?: File; text?: string }) => {
      setAddBusy(true);
      setAddError(null);
      try {
        await addCollection(kind, payload);
        await refresh();
      } catch (e) {
        setAddError(e instanceof Error ? e.message : "Add failed");
        throw e;
      } finally {
        setAddBusy(false);
      }
    },
    [refresh],
  );

  const onDelete = useCallback(
    async (name: string) => {
      try {
        await deleteCollection(name);
        await refresh();
      } catch (e) {
        setAddError(e instanceof Error ? e.message : "Delete failed");
      }
    },
    [refresh],
  );

  const onToggle = useCallback((name: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }, []);

  const onSend = useCallback(
    async (query: string) => {
      const userMsg: ChatMessage = { id: newId(), role: "user", content: query };
      setMessages((prev) => [...prev, userMsg]);
      await send([...selected], query);
    },
    [selected, send],
  );

  // Append the assistant message when the run finishes
  useEffect(() => {
    if (run.status === "done" && run.result) {
      const r = run.result;
      setMessages((prev) => [
        ...prev,
        { id: newId(), role: "assistant", content: r.answer, result: r },
      ]);
    }
  }, [run.status, run.result]);

  return (
    <div className="flex h-screen w-screen overflow-hidden">
      <LeftPanel
        collections={collections}
        selected={selected}
        onToggle={onToggle}
        onAdd={onAdd}
        onDelete={onDelete}
        busy={addBusy}
      />
      <ChatPanel
        messages={messages}
        onSend={onSend}
        isRunning={isRunning}
        hasCollections={selected.size > 0}
      />
      <RightPanel
        result={run.result}
        loading={isRunning}
        error={run.errorMessage ?? addError}
      />
    </div>
  );
}
