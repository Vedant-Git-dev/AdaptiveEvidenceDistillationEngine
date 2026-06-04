"""Workflow Compiler node - Pure Python decision engine."""

from typing import Literal

from aede.state import AEDEState, ReasoningDepth


# Thresholds from config
DEFAULT_COVERAGE_TARGET = 0.8
DEFAULT_REDUNDANCY_THRESHOLD = 0.4
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_MAX_K = 16

# Direct answer thresholds: skip compressor if evidence is good enough
DIRECT_ANSWER_COVERAGE = 0.85
DIRECT_ANSWER_CONFIDENCE = 0.7
DIRECT_ANSWER_REDUNDANCY = 0.2


Decision = Literal[
    "retrieve_more",
    "compress",
    "answer",
    "max_retrieval_reached",
    "direct_answer",
    "llama_with_compress",
    "deep_reasoning",
]


def compiler_decision(
    state: AEDEState,
    coverage_target: float = DEFAULT_COVERAGE_TARGET,
    redundancy_threshold: float = DEFAULT_REDUNDANCY_THRESHOLD,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    max_k: int = DEFAULT_MAX_K,
) -> Decision:
    """
    Compiler decision logic - determines the next step in the workflow.

    Priority-based decision making:
    1. If at max retrieval k -> signal max_retrieval_reached (proceed to answer)
    2. If coverage < target AND missing_parts_core is non-empty -> retrieve more
    3. If redundancy > threshold -> compress (deduplicate)
    4. If confidence < threshold -> retrieve more (low confidence signal)
    5. If required_reasoning == "deep" -> deep_reasoning (compressor + Gemini)
    6. If required_reasoning in {"none","light"} and quality signals are good
       enough:
         - "none"  -> direct_answer (llama, skip compressor)
         - "light" -> llama_with_compress (compressor + llama)
    7. Else -> answer (legacy Gemini path)

    Args:
        state: Current AEDE state
        coverage_target: Target coverage before answering (default: 0.8)
        redundancy_threshold: Redundancy above which we compress (default: 0.4)
        confidence_threshold: Confidence below which we retrieve more (default: 0.5)
        max_k: Maximum retrieval count (default: 32)

    Returns:
        Decision: One of the Decision literals.
    """
    coverage = state.get("coverage", 0.0)
    redundancy = state.get("redundancy", 0.0)
    confidence = state.get("confidence", 0.0)
    missing_parts_core = state.get("missing_parts_core", [])
    max_reached = state.get("max_retrieval_reached", False)
    current_top_k = state.get("current_top_k", 4)
    required_reasoning: ReasoningDepth = state.get("required_reasoning", "deep")

    # Case 1: At max retrieval - signal it and proceed to answer
    # This allows answering even with imperfect coverage
    if max_reached:
        return "max_retrieval_reached"

    # Case 2: Need more evidence for core concepts
    if coverage < coverage_target or len(missing_parts_core) > 0:
        return "retrieve_more"

    # Case 3: Too much redundancy, compress
    if redundancy > redundancy_threshold:
        return "compress"

    # Case 4: Low confidence needs more evidence
    if confidence < confidence_threshold:
        return "retrieve_more"

    # Case 5: Routing based on reasoning depth (analyzer signal).
    # "deep" -> compressor + large model (Gemini).
    if required_reasoning == "deep":
        return "deep_reasoning"

    # Case 6: "none" or "light" can go to the small (llama) model,
    # provided the evidence quality itself is good enough to skip
    # the compressor ("none") or only needs a light compression pass ("light").
    if required_reasoning == "none":
        return "direct_answer"

    if required_reasoning == "light":
        return "llama_with_compress"

    # Case 7: Fallback - default to legacy "answer" (Gemini).
    return "answer"


def workflow_compiler(state: AEDEState) -> AEDEState:
    """
    Main compiler node function for LangGraph.

    Decides next workflow step based on quality signals.
    Returns updated state with workflow_path entry and updated quality signals
    (missing_parts_core recalculated).

    Args:
        state: Current AEDE state

    Returns:
        Updated state with workflow_path entry
    """
    from aede.nodes.analyzer import _calculate_missing_parts_core

    decision = compiler_decision(state)

    # Recalculate missing_parts_core if not present
    if "missing_parts_core" not in state or state["missing_parts_core"] is None:
        missing_parts = state.get("missing_parts", [])
        query_core_concepts = state.get("query_core_concepts", [])
        answered_parts = state.get("answered_parts", [])
        state["missing_parts_core"] = _calculate_missing_parts_core(
            missing_parts, query_core_concepts, answered_parts
        )

    workflow_path = state.get("workflow_path", [])
    workflow_path = workflow_path + [f"compile({decision})"]

    return {
        **state,
        "workflow_path": workflow_path,
    }