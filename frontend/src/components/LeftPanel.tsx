"use client";

import { useEffect, useRef, useState } from "react";
import { FileText, Upload, Loader2, Plus, Trash2, MessageSquare, Bot } from "lucide-react";
import { cn } from "@/lib/cn";

export type AddKind = "pdf" | "chat" | "agent";

export type CollectionItem = {
  name: string;            // "pdf_FY26" or "paste_Q4-chat"
  display_name: string;    // "FY26" or "Q4-chat"
  type: "pdf" | "paste";
  kind?: "chat" | "agent"; // for paste collections
  items: number;
};

type Props = {
  collections: CollectionItem[];
  selected: Set<string>;
  onToggle: (name: string) => void;
  onAdd: (kind: AddKind, payload: { name: string; file?: File; text?: string }) => Promise<void>;
  onDelete: (name: string) => Promise<void>;
  busy: boolean;
};

export function LeftPanel({ collections, selected, onToggle, onAdd, onDelete, busy }: Props) {
  const [addKind, setAddKind] = useState<AddKind | null>(null);
  const [addName, setAddName] = useState("");
  const [addText, setAddText] = useState("");
  const [addFile, setAddFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  const pdfs = collections.filter((c) => c.type === "pdf");
  const chats = collections.filter((c) => c.type === "paste" && c.kind === "chat");
  const agents = collections.filter((c) => c.type === "paste" && c.kind === "agent");

  const reset = () => {
    setAddKind(null);
    setAddName("");
    setAddText("");
    setAddFile(null);
    setError(null);
  };

  const submitAdd = async () => {
    if (!addKind) return;
    if (!addName.trim()) {
      setError("Name is required");
      return;
    }
    if (addKind === "pdf" && !addFile) {
      setError("Pick a PDF file");
      return;
    }
    if ((addKind === "chat" || addKind === "agent") && !addText.trim()) {
      setError("Paste some text");
      return;
    }
    setError(null);
    try {
      await onAdd(addKind, {
        name: addName.trim(),
        file: addFile ?? undefined,
        text: addText.trim() || undefined,
      });
      reset();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to add");
    }
  };

  return (
    <aside className="flex h-full w-80 shrink-0 flex-col gap-4 overflow-y-auto border-r border-surface-200 bg-surface-50 p-4 dark:border-surface-800 dark:bg-surface-900">
      <header>
        <h1 className="text-sm font-semibold tracking-wide text-surface-900 dark:text-surface-100">
          AEDE
        </h1>
        <p className="text-[10px] uppercase tracking-wider text-surface-500">
          Adaptive Evidence-Driven Extraction
        </p>
      </header>

      {/* Add-source radio + conditional input */}
      <section className="card space-y-3">
        <h2 className="text-xs font-semibold tracking-wide text-surface-500 uppercase">Add source</h2>

        <div className="flex flex-col gap-1.5">
          <RadioRow
            icon={<FileText className="h-3.5 w-3.5" />}
            label="PDF"
            active={addKind === "pdf"}
            onClick={() => setAddKind("pdf")}
          />
          <RadioRow
            icon={<MessageSquare className="h-3.5 w-3.5" />}
            label="Chat history"
            active={addKind === "chat"}
            onClick={() => setAddKind("chat")}
          />
          <RadioRow
            icon={<Bot className="h-3.5 w-3.5" />}
            label="Agent conversation"
            active={addKind === "agent"}
            onClick={() => setAddKind("agent")}
          />
        </div>

        {addKind && (
          <div className="space-y-2 border-t border-surface-200 pt-3 dark:border-surface-800">
            <input
              type="text"
              placeholder="Collection name (e.g. FY26-Policy)"
              value={addName}
              onChange={(e) => setAddName(e.target.value)}
              className="w-full rounded-md border border-surface-300 bg-white px-2 py-1.5 text-xs outline-none focus:border-accent-500 dark:border-surface-700 dark:bg-surface-800"
            />

            {addKind === "pdf" ? (
              <FilePicker file={addFile} onFile={setAddFile} />
            ) : (
              <textarea
                placeholder={
                  addKind === "chat"
                    ? "Paste chat log…"
                    : "Paste agent output…"
                }
                value={addText}
                onChange={(e) => setAddText(e.target.value)}
                rows={4}
                className="w-full resize-none rounded-md border border-surface-300 bg-white px-2 py-1.5 font-mono text-xs outline-none focus:border-accent-500 dark:border-surface-700 dark:bg-surface-800"
              />
            )}

            {error && <p className="text-xs text-red-500">{error}</p>}

            <div className="flex gap-2">
              <button
                onClick={submitAdd}
                disabled={busy}
                className="inline-flex flex-1 items-center justify-center gap-1.5 rounded-md bg-accent-500 px-3 py-1.5 text-xs font-medium text-white shadow-sm transition hover:bg-accent-600 disabled:opacity-50"
              >
                {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Plus className="h-3 w-3" />}
                Save collection
              </button>
              <button
                onClick={reset}
                className="rounded-md border border-surface-300 px-3 py-1.5 text-xs text-surface-500 hover:text-surface-900 dark:border-surface-700 dark:hover:text-white"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </section>

      {/* Stored collections, grouped, with checkboxes */}
      <CollectionsGroup
        title="PDFs"
        icon={<FileText className="h-3.5 w-3.5" />}
        items={pdfs}
        selected={selected}
        onToggle={onToggle}
        onDelete={onDelete}
      />
      <CollectionsGroup
        title="Chat history"
        icon={<MessageSquare className="h-3.5 w-3.5" />}
        items={chats}
        selected={selected}
        onToggle={onToggle}
        onDelete={onDelete}
      />
      <CollectionsGroup
        title="Agent conversation"
        icon={<Bot className="h-3.5 w-3.5" />}
        items={agents}
        selected={selected}
        onToggle={onToggle}
        onDelete={onDelete}
      />
    </aside>
  );
}

function RadioRow({
  icon, label, active, onClick,
}: {
  icon: React.ReactNode;
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex items-center gap-2 rounded-md px-2 py-1.5 text-xs transition",
        active
          ? "bg-accent-500/10 text-accent-700 dark:text-accent-300"
          : "text-surface-600 hover:bg-surface-100 dark:text-surface-400 dark:hover:bg-surface-800",
      )}
    >
      <span
        className={cn(
          "flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-full border-2",
          active ? "border-accent-500 bg-accent-500" : "border-surface-300 dark:border-surface-600",
        )}
      >
        {active && <span className="h-1 w-1 rounded-full bg-white" />}
      </span>
      {icon}
      <span className="font-medium">{label}</span>
    </button>
  );
}

function FilePicker({ file, onFile }: { file: File | null; onFile: (f: File | null) => void }) {
  const ref = useRef<HTMLInputElement | null>(null);
  return (
    <div
      onClick={() => ref.current?.click()}
      className="flex cursor-pointer items-center justify-center gap-2 rounded-md border-2 border-dashed border-surface-300 px-3 py-2 text-xs text-surface-500 hover:border-accent-400 dark:border-surface-700"
    >
      <Upload className="h-3.5 w-3.5" />
      <span>{file ? file.name : "Drop or click to upload PDF"}</span>
      <input
        ref={ref}
        type="file"
        accept="application/pdf"
        className="hidden"
        onChange={(e) => onFile(e.target.files?.[0] ?? null)}
      />
    </div>
  );
}

function CollectionsGroup({
  title, icon, items, selected, onToggle, onDelete,
}: {
  title: string;
  icon: React.ReactNode;
  items: CollectionItem[];
  selected: Set<string>;
  onToggle: (name: string) => void;
  onDelete: (name: string) => Promise<void>;
}) {
  if (items.length === 0) return null;
  return (
    <section className="card space-y-2">
      <h2 className="flex items-center gap-1.5 text-xs font-semibold tracking-wide text-surface-500 uppercase">
        {icon}
        {title}
        <span className="ml-auto text-[10px] text-surface-400">{items.length}</span>
      </h2>
      <ul className="flex flex-col gap-1">
        {items.map((c) => (
          <CollectionRow
            key={c.name}
            c={c}
            checked={selected.has(c.name)}
            onToggle={() => onToggle(c.name)}
            onDelete={() => onDelete(c.name)}
          />
        ))}
      </ul>
    </section>
  );
}

function CollectionRow({
  c, checked, onToggle, onDelete,
}: {
  c: CollectionItem;
  checked: boolean;
  onToggle: () => void;
  onDelete: () => void;
}) {
  return (
    <li className="group flex items-center gap-2 rounded-md px-1.5 py-1 text-xs hover:bg-surface-100 dark:hover:bg-surface-800">
      <input
        type="checkbox"
        checked={checked}
        onChange={onToggle}
        className="h-3.5 w-3.5 cursor-pointer rounded border-surface-300 text-accent-500 focus:ring-accent-500 dark:border-surface-600"
      />
      <button
        onClick={onToggle}
        className="flex-1 truncate text-left font-medium text-surface-700 dark:text-surface-200"
      >
        {c.display_name}
      </button>
      <span className="font-mono text-[10px] text-surface-500">{c.items}</span>
      <button
        onClick={onDelete}
        className="opacity-0 transition group-hover:opacity-100"
        aria-label="Delete collection"
      >
        <Trash2 className="h-3 w-3 text-surface-400 hover:text-red-500" />
      </button>
    </li>
  );
}
