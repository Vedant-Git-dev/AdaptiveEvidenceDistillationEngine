"""Node 1: Focused Retriever - Pure retrieval, no LLM."""
from __future__ import annotations

from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from aede.state import AEDEState
from aede.config import settings


# Initialize ChromaDB client
_chroma_client: Optional[chromadb.PersistentClient] = None


def get_chroma_client() -> chromadb.PersistentClient:
    """Get or create ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        settings.retrieval.persist_directory.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=str(settings.retrieval.persist_directory),
            settings=ChromaSettings(
                anonymized_telemetry=False,
            ),
        )
    return _chroma_client


def get_or_create_vectorstore():
    """Get or create the collection with embeddings."""
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=settings.retrieval.collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def focused_retriever(state: AEDEState) -> AEDEState:
    """
    Node 1: Focused Retrieval

    - Pure retrieval, no LLM
    - Starts small: k=4 chunks
    - Binary growth handled by retrieve_more node

    Returns:
        Updated state with documents and current_top_k
    """
    query = state["query"]
    k = 4  # Default to 4 if not set

    # Get documents from vector store
    collection = get_or_create_vectorstore()

    # Query for similar documents
    results = collection.query(
        query_texts=[query],  
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]

    # Build workflow path entry
    workflow_path = state.get("workflow_path", [])
    workflow_path = workflow_path + ["focused_retriever"]

    return {
        "documents": documents,
        "current_top_k": k,
        "workflow_path": workflow_path
    }


def add_documents(documents: list[str], ids: Optional[list[str]] = None, metadatas: Optional[list[dict]] = None) -> int:
    """
    Add documents to the vector store.

    Args:
        documents: List of document texts
        ids: Optional list of IDs (auto-generated if not provided)
        metadatas: Optional list of metadata dicts
    """
    from sentence_transformers import SentenceTransformer

    collection = get_or_create_vectorstore()

    # Generate embeddings
    embedding_model = SentenceTransformer(settings.retrieval.embedding_model)
    embeddings = embedding_model.encode(documents).tolist()

    # Generate IDs if not provided
    if ids is None:
        ids = [f"doc_{i}" for i in range(len(documents))]

    # Add to collection
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        # metadatas=metadatas or [{}] * len(documents),
    )

    return len(documents)


class MockCollection:
    """Mock ChromaDB collection for testing."""

    def __init__(self, documents: Optional[list[str]] = None):
        self.documents = documents or []
        self.query_calls = []

    def query(self, query_texts: list[str], n_results: int, include: Optional[list[str]] = None):
        self.query_calls.append({
            "query_texts": query_texts,
            "n_results": n_results,
            "include": include,
        })
        return {
            "documents": [self.documents[:n_results]],
            "metadatas": [[]],
            "distances": [[0.0] * min(n_results, len(self.documents))],
        }

    def add(self, ids: list[str], documents: list[str], embeddings: list, metadatas: list[dict]):
        self.documents.extend(documents)