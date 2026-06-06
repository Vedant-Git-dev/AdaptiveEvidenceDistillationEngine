"""Adapters that turn host-supplied input into `list[ContextItem]`.

The adapters are the only place AEDE learns about specific input shapes
(PDF bytes, pasted text). Everything downstream operates on `ContextItem`.
"""
from aede.adapters.text_chunker import chunk_text
from aede.adapters.pdf import pdf_to_context_items

__all__ = ["chunk_text", "pdf_to_context_items"]
