"""Node 1: Focused Retriever - Pure retrieval, no LLM.

Also exposes per-collection helpers so the API can keep one Chroma collection
per uploaded PDF.
"""
from __future__ import annotations

import re
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


def _sanitize_collection_name(name: str) -> str:
    """Chroma collection names: 3-63 chars, [a-zA-Z0-9_-], must start/end alnum."""
    s = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    s = re.sub(r"_+", "_", s).strip("_-")
    if not s or not s[0].isalnum():
        s = "c_" + s
    s = s[:63]
    if len(s) < 3:
        s = (s + "___")[:3]
    return s


def get_or_create_vectorstore(collection_name: Optional[str] = None):
    """Get or create a Chroma collection. Defaults to the global collection."""
    client = get_chroma_client()
    name = collection_name or settings.retrieval.collection_name
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def collection_name_for(filename: str) -> str:
    """Stable, sanitized collection name for a given PDF filename."""
    stem = filename.rsplit("/", 1)[-1]
    stem = stem.rsplit(".", 1)[0] if "." in stem else stem
    return _sanitize_collection_name(f"pdf_{stem}")


def list_collections() -> list[dict]:
    """List all AEDE-managed PDF collections (excludes the default scratch one)."""
    client = get_chroma_client()
    out = []
    for c in client.list_collections():
        name = c.name
        if not name.startswith("pdf_"):
            continue
        try:
            count = c.count()
        except Exception:
            count = 0
        out.append({"name": name, "embeddings": count})
    return out


def delete_collection(name: str) -> None:
    client = get_chroma_client()
    try:
        client.delete_collection(name)
    except Exception:
        pass


def focused_retriever(
    state: AEDEState,
    collection_name: Optional[str] = None,
) -> AEDEState:
    """Node 1: Focused retrieval against a specific collection (if given).

    The collection name is read from the explicit argument first, then from
    state["collection_name"], then falls back to the default. The graph
    doesn't pass an explicit kwarg, so state is the canonical channel.
    """
    query = state["query"]
    k = 4

    cn = collection_name or state.get("collection_name")
    collection = get_or_create_vectorstore(cn)

    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]

    workflow_path = state.get("workflow_path", [])
    workflow_path = workflow_path + ["focused_retriever"]

    return {
        "documents": documents,
        "current_top_k": k,
        "workflow_path": workflow_path,
    }


def retrieve_more(
    state: AEDEState,
    collection_name: Optional[str] = None,
) -> AEDEState:
    """Node 5: incremental retrieval, binary growth k=4→8→16→MAX."""
    current_k = state["current_top_k"]
    if current_k == 4:
        new_k = 8
    elif current_k == 8:
        new_k = 16
    else:
        new_k = settings.retrieval.max_k

    query = state["query"]
    cn = collection_name or state.get("collection_name")
    collection = get_or_create_vectorstore(cn)
    results = collection.query(
        query_texts=[query],
        n_results=new_k,
        include=["documents", "metadatas", "distances"],
    )
    documents = results.get("documents", [[]])[0]

    workflow_path = state.get("workflow_path", [])
    workflow_path = workflow_path + [f"retrieve_more(k={new_k})"]

    max_reached = new_k >= settings.retrieval.max_k

    return {
        "documents": documents,
        "current_top_k": new_k,
        "workflow_path": workflow_path,
        "max_retrieval_reached": max_reached,
        "retrieve_more_count": state.get("retrieve_more_count", 0) + 1,
    }


def add_documents(
    documents: list[str],
    ids: Optional[list[str]] = None,
    metadatas: Optional[list[dict]] = None,
    collection_name: Optional[str] = None,
) -> int:
    """Add documents to a Chroma collection (per-PDF if collection_name given)."""
    from sentence_transformers import SentenceTransformer

    collection = get_or_create_vectorstore(collection_name)
    embedding_model = SentenceTransformer(settings.retrieval.embedding_model)
    embeddings = embedding_model.encode(documents).tolist()

    if ids is None:
        ids = [f"doc_{i}" for i in range(len(documents))]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
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
