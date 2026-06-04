"""Ingest a PDF into the AEDE vector store.

Usage:
    python3 scripts/ingest_pdf.py path/to/file.pdf
    python3 scripts/ingest_pdf.py path/to/file.pdf --chunk-size 1000 --overlap 200
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# Make `aede` importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pdfplumber

from aede.nodes.retrieval import add_documents


def extract_text(pdf_path: Path) -> str:
    """Extract text from all pages of the PDF, joined with double newlines."""
    pages: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text)
    return "\n\n".join(pages)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sliding-window chunking by character count.

    Whitespace is normalised first so chunks don't end mid-blank-line.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # Normalise whitespace: collapse runs of blank lines, strip per-line.
    cleaned = "\n".join(line.strip() for line in text.splitlines())
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    cleaned = cleaned.strip()

    if not cleaned:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(cleaned), step):
        chunk = cleaned[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(cleaned):
            break
    return chunks


def make_ids(pdf_path: Path, chunks: list[str]) -> list[str]:
    """Stable IDs: <filename-stem>-<sha1[:8]-of-chunk>-<index>."""
    stem = pdf_path.stem.replace(" ", "_")
    ids = []
    for i, chunk in enumerate(chunks):
        digest = hashlib.sha1(chunk.encode("utf-8")).hexdigest()[:8]
        ids.append(f"{stem}-{digest}-{i}")
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a PDF into the AEDE vector store.")
    parser.add_argument("pdf", type=Path, help="Path to the PDF file")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Characters per chunk (default: 1000)")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap between chunks (default: 200)")
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"Error: {args.pdf} not found", file=sys.stderr)
        return 1
    if args.pdf.suffix.lower() != ".pdf":
        print(f"Error: {args.pdf} is not a .pdf", file=sys.stderr)
        return 1

    print(f"Extracting text from {args.pdf} ...")
    text = extract_text(args.pdf)
    if not text.strip():
        print("Error: no extractable text found (scanned PDF?)", file=sys.stderr)
        return 1
    print(f"  {len(text):,} characters extracted")

    print(f"Chunking (size={args.chunk_size}, overlap={args.overlap}) ...")
    chunks = chunk_text(text, args.chunk_size, args.overlap)
    print(f"  {len(chunks)} chunks")

    ids = make_ids(args.pdf, chunks)
    print(f"Adding to vector store ...")
    n = add_documents(chunks, ids=ids)
    print(f"  added {n} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
