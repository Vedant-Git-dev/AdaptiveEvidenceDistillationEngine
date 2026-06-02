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
)


def _route_from_compiler(state: AEDEState) -> str:
    """
    Route from compiler based on decision.

    Returns:
        "retrieve_more" -> continue to retrieve_more node
        "compress" -> continue to compressor node
        "answer" -> go directly to reasoner
        "max_retrieval_reached" -> go directly to reasoner
    """
    from aede.nodes.compiler import compiler_decision

    decision = compiler_decision(state)
    return decision


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
           +-- "compress" -> evidence_compressor -> final_reasoner
           |
           +-- "answer" / "max_retrieval_reached" -> final_reasoner -> END

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
            "answer": "reason",
            "max_retrieval_reached": "reason",
        },
    )

    # After retrieve_more, loop back to extract
    graph.add_edge("retrieve_more", "extract")

    # After compress, go to reasoner
    graph.add_edge("compress", "reason")

    # Reasoner is terminal
    graph.add_edge("reason", END)

    return graph.compile()


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