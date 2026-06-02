"""Shared state schema for AEDE LangGraph."""

from typing import TypedDict, NotRequired
from typing_extensions import Annotated
import operator


class Fact(TypedDict):
    """A single extracted fact with source attribution."""
    claim: str
    quote: str
    chunk_id: int


class AEDEState(TypedDict):
    """State passed through the AEDE pipeline."""

    # === Inputs ===
    query: str
    query_core_concepts: NotRequired[list[str]]

    # === Retrieval ===
    current_top_k: int
    documents: NotRequired[list[str]]

    # === Evidence pipeline ===
    facts: NotRequired[list[Fact]]

    # === Compression ===
    compressed_evidence: NotRequired[list[str]]

    # === Quality signals (from Node 3) ===
    answered_parts: NotRequired[list[str]]
    missing_parts: NotRequired[list[str]]
    missing_parts_core: NotRequired[list[str]]  # missing_parts ∩ query_core_concepts
    coverage: NotRequired[float]  # 0.0 - 1.0
    redundancy: NotRequired[float]  # 0.0 - 1.0
    confidence: NotRequired[float]  # 0.0 - 1.0

    # === Decision tracking ===
    workflow_path: NotRequired[list[str]]  # ["retrieve", "extract", "analyze", ...]
    coverage_history: NotRequired[list[float]]  # [0.31, 0.57, 0.81, 0.90]
    token_usage: NotRequired[dict[str, int]]  # {"gemma_input": 5000, "gemma_output": 200, ...}

    # === Outputs ===
    answer: NotRequired[str]
    max_retrieval_reached: NotRequired[bool]  # Flag when k=MAX and coverage < 0.8
    error: NotRequired[str | None]


def create_initial_state(query: str) -> AEDEState:
    """Create initial state for a new query."""
    return AEDEState(
        query=query,
        query_core_concepts=[],
        current_top_k=4,
        documents=[],
        facts=[],
        compressed_evidence=[],
        answered_parts=[],
        missing_parts=[],
        missing_parts_core=[],
        coverage=0.0,
        redundancy=0.0,
        confidence=0.0,
        workflow_path=["start"],
        coverage_history=[],
        token_usage={},
        answer="",
        max_retrieval_reached=False,
        error=None,
    )