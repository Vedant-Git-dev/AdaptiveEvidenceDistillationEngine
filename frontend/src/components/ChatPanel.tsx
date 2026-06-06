"use client";

import { useEffect, useRef, useState } from "react";
import { Send, Loader2, Bot, User } from "lucide-react";
import { cn } from "@/lib/cn";
import type { ChatMessage, OptimizeResponse } from "@/lib/types";

type Props = {
  messages: ChatMessage[];
  onSend: (q: string) => Promise<void> | void;
  isRunning: boolean;
  hasCollections: boolean;
};

export function ChatPanel({ messages, onSend, isRunning, hasCollections }: Props) {
  const [input, setInput] = useState("");
  const scrollerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = scrollerRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, isRunning]);

  const submit = async () => {
    const q = input.trim();
    if (!q || isRunning || !hasCollections) return;
    setInput("");
    await onSend(q);
  };

  return (
    <main className="flex h-full min-w-0 flex-1 flex-col bg-white dark:bg-surface-950">
      <div ref={scrollerRef} className="flex-1 overflow-y-auto px-6 py-8">
        {messages.length === 0 ? (
          <EmptyState hasCollections={hasCollections} />
        ) : (
          <div className="mx-auto flex max-w-3xl flex-col gap-6">
            {messages.map((m) => (
              <Bubble key={m.id} message={m} />
            ))}
            {isRunning && <ThinkingBubble />}
          </div>
        )}
      </div>

      <div className="border-t border-surface-200 bg-surface-50 px-6 py-4 dark:border-surface-800 dark:bg-surface-900">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void submit();
          }}
          className="mx-auto flex max-w-3xl items-end gap-2"
        >
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void submit();
              }
            }}
            placeholder={
              hasCollections
                ? "Ask a question across the selected collections…"
                : "Add a collection on the left, then select it and ask…"
            }
            rows={1}
            disabled={!hasCollections}
            className="min-h-[44px] flex-1 resize-none rounded-lg border border-surface-300 bg-white px-3 py-2.5 text-sm shadow-sm outline-none focus:border-accent-500 focus:ring-2 focus:ring-accent-500/20 disabled:opacity-60 dark:border-surface-700 dark:bg-surface-800"
          />
          <button
            type="submit"
            disabled={!input.trim() || isRunning || !hasCollections}
            className="inline-flex h-11 w-11 items-center justify-center rounded-lg bg-accent-500 text-white shadow-sm transition hover:bg-accent-600 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {isRunning ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          </button>
        </form>
      </div>
    </main>
  );
}

function Bubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  return (
    <div className={cn("flex gap-3", isUser ? "justify-end" : "justify-start")}>
      {!isUser && (
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-accent-500/10 text-accent-600 dark:text-accent-400">
          <Bot className="h-4 w-4" />
        </div>
      )}
      <div
        className={cn(
          "max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm",
          isUser
            ? "bg-accent-500 text-white"
            : "border border-surface-200 bg-white dark:border-surface-800 dark:bg-surface-900",
        )}
      >
        <p className="whitespace-pre-wrap">{message.content}</p>
      </div>
      {isUser && (
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-surface-200 text-surface-700 dark:bg-surface-800 dark:text-surface-300">
          <User className="h-4 w-4" />
        </div>
      )}
    </div>
  );
}

function ThinkingBubble() {
  return (
    <div className="flex gap-3">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-accent-500/10 text-accent-600 dark:text-accent-400">
        <Bot className="h-4 w-4" />
      </div>
      <div className="flex items-center gap-2 rounded-2xl border border-surface-200 bg-white px-4 py-3 text-sm text-surface-500 dark:border-surface-800 dark:bg-surface-900">
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        <span>Running AEDE…</span>
      </div>
    </div>
  );
}

function EmptyState({ hasCollections }: { hasCollections: boolean }) {
  return (
    <div className="mx-auto flex h-full max-w-2xl flex-col items-center justify-center text-center">
      <Bot className="mb-4 h-10 w-10 text-accent-500" />
      <h2 className="text-lg font-semibold">
        {hasCollections ? "Ask across your selected collections" : "Add a collection to get started"}
      </h2>
      <p className="mt-1 text-sm text-surface-500">
        {hasCollections
          ? "Type below. AEDE picks the smallest context the frontier model needs to answer, and the right panel shows what got sent vs. raw Gemini."
          : "On the left, add a PDF or paste some text. Give it a name, save it, then check the box and ask."}
      </p>
    </div>
  );
}
