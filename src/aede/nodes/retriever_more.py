"""Node 5: Incremental Retriever - Binary growth strategy."""

from aede.state import AEDEState
from aede.config import settings
from aede.nodes.retrieval import get_or_create_vectorstore


def retrieve_more(state: AEDEState) -> AEDEState:
    """
    Node 5: Incremental Retrieval

    Binary growth strategy: k=4→8→16→MAX (not k+=4)

    Returns:
        Updated state with more documents and new k value
    """
    current_k = state["current_top_k"]

    # Binary growth
    if current_k == 4:
        new_k = 8
    elif current_k == 8:
        new_k = 16
    else:
        new_k = settings.retrieval.max_k  # MAX (32)

    query = state["query"]

    # Get documents from vector store
    collection = get_or_create_vectorstore()
    results = collection.query(
        query_texts=[query],
        n_results=new_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]

    # Track workflow path
    workflow_path = state.get("workflow_path", [])
    workflow_path = workflow_path + [f"retrieve_more(k={new_k})"]

    # Check if we've reached max retrieval
    max_reached = new_k >= settings.retrieval.max_k

    return {
        "documents": documents,
        "current_top_k": new_k,
        "workflow_path": workflow_path,
        "max_retrieval_reached": max_reached,
    }