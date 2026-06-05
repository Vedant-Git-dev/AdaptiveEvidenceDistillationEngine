"""Display metadata for the models and nodes in the AEDE pipeline.

The model name shown in the right panel comes from the actual configured
pipeline model in `aede.config.settings` — so changing the model on the
backend automatically updates the UI without any code change here.

This module is the single place that knows which model runs each step.
"""

from __future__ import annotations

from aede.config import settings


# Pipeline model that runs extract / analyze / compress / small_reasoner.
# Pulled live from settings so the UI label always matches what was actually
# called. If we ever switch to multiple models per node, this becomes a
# node→model map.
_PIPELINE_MODEL = settings.models.pipeline_model
_GEMINI_MODEL = settings.models.gemini_reasoner_model


def _short(name: str) -> str:
    """Drop the vendor prefix for compact display: 'llama-3.3-70B-versatile' → 'Llama 3.3 70B'."""
    n = name.strip()
    if "-" in n:
        parts = n.split("-")
        # Take the meaningful part (skip generic family prefix if present)
        # e.g. "llama-3.3-70b-versatile" → "Llama 3.3 70B"
        # e.g. "gemini-2.5-flash" → "Gemini 2.5 Flash"
        if parts[0].lower() in {"llama", "gemini", "gpt", "claude"}:
            cap = " ".join(p.capitalize() for p in parts)
            # Normalize "70b" → "70B"
            cap = cap.replace(" 70b", " 70B").replace(" 8b", " 8B").replace(" 405b", " 405B")
            return cap
    return n


_PIPELINE_SHORT = _short(_PIPELINE_MODEL)
_GEMINI_SHORT = _short(_GEMINI_MODEL)


# Short, human-friendly labels for the model that runs each step.
STEP_MODELS: dict[str, str] = {
    "extract_concepts": "Keyword extraction",
    "focused_retriever": "ChromaDB",
    "retrieve": "ChromaDB",
    "retrieve_more": "ChromaDB",
    "extract": _PIPELINE_SHORT,
    "analyze": _PIPELINE_SHORT,
    "compress": _PIPELINE_SHORT,
    "compile": "Compiler",
    "reason": _GEMINI_SHORT,
    "small_reasoner": _PIPELINE_SHORT,
}


# Full names used in tooltips / detail rows.
STEP_MODEL_FULL: dict[str, str] = {
    "focused_retriever": "ChromaDB + sentence-transformers",
    "retrieve_more": "ChromaDB + sentence-transformers",
    "extract": f"{_PIPELINE_MODEL} (Groq)",
    "analyze": f"{_PIPELINE_MODEL} (Groq)",
    "compress": f"{_PIPELINE_MODEL} (Groq)",
    "compile": "Pure Python decision engine",
    "reason": f"{_GEMINI_MODEL} (Google)",
    "small_reasoner": f"{_PIPELINE_MODEL} (Groq)",
}


def model_for(node_name: str) -> str:
    """Short model label for a node name. Falls back gracefully."""
    return STEP_MODELS.get(node_name, "—")
