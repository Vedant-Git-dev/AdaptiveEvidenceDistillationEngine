"""Node 1: Focused Retriever - Pure retrieval, no LLM.

Chroma persists to disk (PersistentClient). The /optimize endpoint creates
a fresh per-request collection for every call and deletes it at the end of
the request, so nothing carries over between requests — the persistence is
just there to avoid loading the embedding index into memory if you want to
inspect it manually during a request.
"""
from __future__ import annotations

from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from aede.state import AEDEState
from aede.config import settings


# Single persistent client, process-wide. Points at backend/data/chroma_db.
_chroma_client: chromadb.PersistentClient | None = None


# Fixed name for the per-request scratch collection. The /optimize endpoint
# recreates it on every call so retrieval always starts empty.
REQUEST_COLLECTION = "aede_request"


def get_chroma_client() -> chromadb.PersistentClient:
    """Get or create the persistent Chroma client."""
    global _chroma_client
    if _chroma_client is None:
        settings.retrieval.persist_directory.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=str(settings.retrieval.persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _chroma_client


def make_request_collection() -> chromadb.Collection:
    """Create a fresh, empty collection for the current /optimize call.

    Any leftover collection with the same name is deleted first so that the
    request always starts from a clean slate. Nothing persists across calls.
    """
    client = get_chroma_client()
    try:
        client.delete_collection(REQUEST_COLLECTION)
    except Exception:
        pass
    return client.get_or_create_collection(
        name=REQUEST_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def drop_request_collection() -> None:
    """Delete the per-request collection. Called by /optimize at the end of
    the request so nothing carries over to the next call."""
    client = get_chroma_client()
    try:
        client.delete_collection(REQUEST_COLLECTION)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Multi-collection pool: query N collections, pool top-k from each
# ---------------------------------------------------------------------------


def query_pool(
    collection_names: list[str],
    query: str,
    per_collection_k: int,
) -> list[str]:
    """Query each named collection with the same query, return the union of
    the top-k docs from each. Used by /optimize when multiple collections
    are selected."""
    client = get_chroma_client()
    pooled: list[str] = []
    for name in collection_names:
        try:
            coll = client.get_collection(name=name)
        except Exception:
            continue
        results = coll.query(
            query_texts=[query],
            n_results=per_collection_k,
            include=["documents"],
        )
        docs = results.get("documents", [[]])[0]
        pooled.extend(docs)
    return pooled


def list_aede_collections() -> list[dict]:
    """List all AEDE-managed collections with item counts.

    Each item: {name, display_name, items, type, kind?}
    - name: internal collection id (e.g. "pdf_FY26-Policy")
    - display_name: the user-supplied name (e.g. "FY26-Policy"); falls back
      to a stripped version of `name` for legacy collections.
    - type: "pdf" | "paste" | "other"
    - kind: "chat" | "agent" (paste only)
    """
    client = get_chroma_client()
    out: list[dict] = []
    for c in client.list_collections():
        name = c.name
        if name == REQUEST_COLLECTION:
            continue
        try:
            count = c.count()
        except Exception:
            count = 0

        # Pull display_name + kind from the collection's metadata, which the
        # add endpoints set at creation time. Legacy collections don't have
        # these — fall back to stripping the prefix.
        meta = c.metadata or {}
        display_name = meta.get("display_name") or _fallback_display_name(name)
        kind = meta.get("source_kind") if name.startswith("paste_") else None

        type_label = (
            "pdf" if name.startswith("pdf_")
            else "paste" if name.startswith("paste_")
            else "other"
        )
        item: dict = {
            "name": name,
            "display_name": display_name,
            "items": count,
            "type": type_label,
        }
        if kind and type_label == "paste":
            item["kind"] = kind
        out.append(item)
    return out


def _fallback_display_name(internal_name: str) -> str:
    """For collections added before display_name was stored in metadata,
    derive something readable by stripping the pdf_/paste_ prefix."""
    for prefix in ("pdf_", "paste_"):
        if internal_name.startswith(prefix):
            return internal_name[len(prefix):]
    return internal_name


def delete_aede_collection(name: str) -> bool:
    """Delete a single named collection. Returns True if it existed."""
    if name == REQUEST_COLLECTION:
        return False
    client = get_chroma_client()
    try:
        client.delete_collection(name)
        return True
    except Exception:
        return False


def get_or_create_vectorstore(collection_name: Optional[str] = None):
    """Back-compat shim. The pipeline always reads from the request collection;
    the argument is accepted and ignored."""
    client = get_chroma_client()
    name = collection_name or REQUEST_COLLECTION
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def focused_retriever(
    state: AEDEState,
    collection_name: Optional[str] = None,
) -> AEDEState:
    """Node 1: Focused retrieval against the per-request Chroma collection."""
    query = state["query"]
    k = 4

    collection = get_or_create_vectorstore(collection_name)

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
    collection = get_or_create_vectorstore(collection_name)
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
    """Embed `documents` with sentence-transformers and write to the per-request
    Chroma collection."""
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
