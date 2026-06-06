"""Sentence-aware semantic chunker + ContextItem factory.

Used by the conversation and agent paste sources. The PDF adapter reuses the
same chunker after page-level text extraction.
"""
from __future__ import annotations

import re
from typing import Iterable

from aede.types import ContextItem


TARGET_CHUNK = 1000          # ~ characters
OVERLAP = 200
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def semantic_chunks(text: str) -> list[str]:
    """Sentence-aware chunker: respects paragraph boundaries, ~1000 char
    target with 200 char overlap. Falls back to sentence splitting on long
    paragraphs.
    """
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buffer = ""

    def flush(buf: str) -> None:
        if buf.strip():
            chunks.append(buf.strip())

    for para in paragraphs:
        if len(para) > TARGET_CHUNK:
            sentences = SENTENCE_END.split(para)
            for sent in sentences:
                if len(buffer) + len(sent) + 1 > TARGET_CHUNK and buffer:
                    flush(buffer)
                    buffer = buffer[-OVERLAP:] + " " + sent
                else:
                    buffer = (buffer + " " + sent).strip()
        else:
            if len(buffer) + len(para) + 2 > TARGET_CHUNK and buffer:
                flush(buffer)
                buffer = buffer[-OVERLAP:] + "\n\n" + para
            else:
                buffer = (buffer + "\n\n" + para).strip()
    flush(buffer)
    return chunks


def _approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars per token). Replace with a real counter
    if precision matters; the AEDE pipeline only uses this for display."""
    return max(1, len(text) // 4)


def chunk_text(
    text: str,
    source_type: str,
    label: str = "paste",
) -> list[ContextItem]:
    """Chunk a plain text blob and return ContextItems."""
    pieces = semantic_chunks(text)
    return [
        ContextItem(
            id=f"{source_type}::{label}::{i}",
            text=piece,
            source_type=source_type,            # type: ignore[arg-type]
            tokens=_approx_tokens(piece),
            metadata={"label": label, "chunk_index": i},
        )
        for i, piece in enumerate(pieces)
    ]


def join_texts(blobs: Iterable[tuple[str, str]]) -> str:
    """Helper: join (label, text) tuples with section markers before chunking.
    Used when one source produces multiple disjoint blobs (e.g. a long PDF)."""
    parts: list[str] = []
    for label, body in blobs:
        if body:
            parts.append(f"--- {label} ---\n{body}")
    return "\n\n".join(parts)
