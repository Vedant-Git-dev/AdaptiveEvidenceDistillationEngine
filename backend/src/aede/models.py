"""Display metadata for the models and nodes in the AEDE pipeline.

Keeps the UI string for each step in one place so the workflow panel can show
"Extract (Llama 8B)" without knowing about the backend's internals.
"""

from aede.config import settings


# Short, human-friendly labels for the model that runs each step.
# `id` is what the right panel shows in the gray pill next to the step name.
STEP_MODELS: dict[str, str] = {
    "extract_concepts": "Keyword extraction",
    "focused_retriever": "ChromaDB",
    "retrieve": "ChromaDB",
    "retrieve_more": "ChromaDB",
    "extract": "Llama 8B",
    "analyze": "Llama 8B",
    "compress": "Llama 8B",
    "compile": "Compiler",
    "reason": "Gemini Flash",
    "small_reasoner": "Llama 8B",
}


# Full names used in tooltips / detail rows.
STEP_MODEL_FULL: dict[str, str] = {
    "focused_retriever": "ChromaDB + sentence-transformers",
    "retrieve_more": "ChromaDB + sentence-transformers",
    "extract": f"{settings.models.pipeline_model} (Groq)",
    "analyze": f"{settings.models.pipeline_model} (Groq)",
    "compress": f"{settings.models.pipeline_model} (Groq)",
    "compile": "Pure Python decision engine",
    "reason": f"{settings.models.gemini_reasoner_model} (Google)",
    "small_reasoner": f"{settings.models.pipeline_model} (Groq)",
}


def model_for(node_name: str) -> str:
    """Short model label for a node name. Falls back gracefully."""
    return STEP_MODELS.get(node_name, "—")
