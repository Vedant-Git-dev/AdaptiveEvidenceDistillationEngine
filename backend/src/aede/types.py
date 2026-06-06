"""Public types for the AEDE context-optimization engine.

These types are the contract between the host application and the AEDE core.
AEDE itself operates on `ContextItem` lists; it does not know how they were
produced, where they came from, or what will be done with the answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Core engine types
# ---------------------------------------------------------------------------

SourceType = Literal["pdf_chunk", "chat_history", "tool_output"]


@dataclass
class ContextItem:
    """A single piece of evidence fed into the AEDE pipeline.

    Items are source-agnostic. The pipeline does not read `source_type` or
    `origin` — those exist for the host application to display provenance.
    """

    id: str
    text: str
    source_type: SourceType
    score: float = 0.0
    tokens: int = 0
    metadata: dict | None = None
    origin: Literal["ephemeral"] = "ephemeral"
    label: str = ""


@dataclass
class TokenBudget:
    """How aggressively AEDE should compress the context for this request."""

    hard_limit: int = 4000
    target: int = 3000
    reserved_for_answer: int = 800


# ---------------------------------------------------------------------------
# HTTP request/response schemas (live here so the engine stays import-clean)
# ---------------------------------------------------------------------------


class OptimizeRequest(BaseModel):
    """One /optimize call. Exactly one of the source payloads is filled."""

    source: Literal["pdf", "conversation", "agent"]
    query: str = Field(..., min_length=1)

    # PDF
    pdf_file: str | None = Field(default=None, description="filename hint only")

    # Conversation paste
    conversation_text: str | None = None

    # Agent paste
    agent_text: str | None = None

    # Budget knobs
    budget_target: int = 3000
    budget_hard_limit: int = 4000


class OptimizeResponse(BaseModel):
    answer: str
    decision: str
    final_tokens: int
    raw_tokens: int
    saved_pct: float
    items_count: int
    source: str
    trace: list[str] = field(default_factory=list)
    coverage: float = 0.0
    workflow_path: list[str] = field(default_factory=list)
