"""Main LangGraph for AEDE pipeline."""

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from aede.state import AEDEState, create_initial_state
from aede.nodes import (
    extract_core_concepts,
    focused_retriever,
    evidence_extractor,
    evidence_analyzer,
    workflow_compiler,
    evidence_compressor,
    retrieve_more,
    final_reasoner,
    small_reasoner,
)


def _route_from_compiler(state: AEDEState) -> str:
    """
    Route from compiler based on decision.

    Returns one of the Decision literals. The conditional edge map in
    build_graph() translates each literal to a destination node.
    """
    from aede.nodes.compiler import compiler_decision

    return compiler_decision(state)


def build_graph() -> StateGraph:
    """
    Build the AEDE LangGraph.

    Flow:
        START -> extract_core_concepts -> focused_retriever
        -> evidence_extractor -> evidence_analyzer
        -> workflow_compiler (decides next step)
           |
           +-- "retrieve_more" -> retrieve_more -> evidence_extractor (loop)
           |
           +-- "compress" -> evidence_compressor -> final_reasoner (Gemini)
           |
           +-- "direct_answer" -> small_reasoner (llama, no compression)
           |
           +-- "llama_with_compress" -> evidence_compressor -> small_reasoner
           |
           +-- "deep_reasoning" -> evidence_compressor -> final_reasoner
           |
           +-- "answer" -> evidence_compressor -> final_reasoner
           |
           +-- "max_retrieval_reached" -> final_reasoner -> END

    Returns:
        Compiled StateGraph ready to run
    """
    # Define the graph
    graph = StateGraph(AEDEState)

    # Add all nodes
    graph.add_node("extract_concepts", extract_core_concepts)
    graph.add_node("retrieve", focused_retriever)
    graph.add_node("extract", evidence_extractor)
    graph.add_node("analyze", evidence_analyzer)
    graph.add_node("compile", workflow_compiler)
    graph.add_node("retrieve_more", retrieve_more)
    graph.add_node("compress", evidence_compressor)
    graph.add_node("reason", final_reasoner)
    graph.add_node("small_reasoner", small_reasoner)

    # Define edges
    graph.set_entry_point("extract_concepts")
    graph.add_edge("extract_concepts", "retrieve")
    graph.add_edge("retrieve", "extract")
    graph.add_edge("extract", "analyze")
    graph.add_edge("analyze", "compile")

    # Conditional edge from compiler
    graph.add_conditional_edges(
        "compile",
        _route_from_compiler,
        {
            "retrieve_more": "retrieve_more",
            "compress": "compress",
            # Both "answer" (legacy fallback) and "deep_reasoning" (new explicit
            # deep path) send the evidence through the compressor on its way
            # to the large model (Gemini). The decision literal is preserved
            # in workflow_path for observability.
            "answer": "compress",
            "deep_reasoning": "compress",
            # New routing based on required_reasoning:
            "direct_answer": "small_reasoner",      # skip compressor -> llama
            "llama_with_compress": "compress",       # compressor -> llama
            "max_retrieval_reached": "reason",
        },
    )

    # After retrieve_more, loop back to extract
    graph.add_edge("retrieve_more", "extract")

    # After compress, branch by the conditional edge below
    graph.add_conditional_edges(
        "compress",
        _route_after_compress,
        {
            "reason": "reason",
            "small_reasoner": "small_reasoner",
        },
    )

    # Terminal edges
    graph.add_edge("reason", END)
    graph.add_edge("small_reasoner", END)

    return graph.compile()


def _route_after_compress(state: AEDEState) -> str:
    """
    Route from the compressor based on the decision recorded in workflow_path.

    The compiler's last decision is recorded in workflow_path as
    "compile(<decision>)". We read it here to decide whether to send the
    compressed evidence to the small model or the large one.

    Returns:
        "reason" -> final_reasoner (Gemini)
        "small_reasoner" -> small_reasoner (llama)
    """
    path = state.get("workflow_path", []) or []
    last_compile = next(
        (step for step in reversed(path) if step.startswith("compile(")),
        "",
    )
    # Format: "compile(<decision>)"
    if last_compile.startswith("compile(") and last_compile.endswith(")"):
        decision = last_compile[len("compile("):-1]
        if decision == "llama_with_compress":
            return "small_reasoner"
    return "reason"


# Default graph instance
_default_graph = None


def get_graph() -> StateGraph.compile:
    """Get or create the default compiled graph."""
    global _default_graph
    if _default_graph is None:
        _default_graph = build_graph()
    return _default_graph


def run(query: str) -> AEDEState:
    """
    Run the AEDE pipeline on a query.

    Args:
        query: User query

    Returns:
        Final state with answer
    """
    graph = get_graph()
    initial_state = create_initial_state(query)
    final_state = graph.invoke(initial_state)
    return final_state


async def run_async(query: str) -> AEDEState:
    """
    Run the AEDE pipeline asynchronously.

    Args:
        query: User query

    Returns:
        Final state with answer
    """
    graph = get_graph()
    initial_state = create_initial_state(query)
    final_state = await graph.ainvoke(initial_state)
    return final_state