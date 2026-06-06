"""PDF adapter: extract text per page, then chunk via the shared chunker."""
from __future__ import annotations

import io

from pypdf import PdfReader

from aede.adapters.text_chunker import _approx_tokens, semantic_chunks
from aede.types import ContextItem


def pdf_to_context_items(file_bytes: bytes, filename: str = "upload.pdf") -> list[ContextItem]:
    """Extract text from a PDF and return one ContextItem per chunk.

    Each item carries the source page in metadata so the right panel can
    render page citations.
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    items: list[ContextItem] = []

    for page_idx, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        if not text.strip():
            continue

        for chunk_idx, chunk in enumerate(semantic_chunks(text)):
            items.append(
                ContextItem(
                    id=f"pdf_chunk::{filename}::p{page_idx + 1}::c{chunk_idx}",
                    text=chunk,
                    source_type="pdf_chunk",
                    tokens=_approx_tokens(chunk),
                    metadata={
                        "filename": filename,
                        "page": page_idx + 1,
                        "chunk_index": chunk_idx,
                    },
                    label=f"{filename} p.{page_idx + 1}",
                )
            )

    return items
