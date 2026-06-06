"""FastAPI sidecar for the AEDE frontend.

Endpoints (persistent, named collections):
  GET    /collections                  — list all stored collections
  POST   /collections/pdf              — add a PDF (multipart: name, file)
  POST   /collections/paste            — add a text blob (form: name, text, kind)
  DELETE /collections/{name}           — drop a collection
  POST   /optimize                     — run AEDE against selected collections

Chroma is persistent. The user names each collection at creation time.
Nothing is wiped automatically — the user deletes what they don't want.
"""
from __future__ import annotations

import os
import re
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from aede.adapters.text_chunker import semantic_chunks
from aede.config import settings
from aede.graph import get_graph
from aede.runner import run_with_timings
from aede.nodes.retrieval import (
    list_aede_collections,
    delete_aede_collection,
    get_chroma_client,
    query_pool,
)
from aede.state import create_initial_state


# ---------------------------------------------------------------------------
# Collection naming
# ---------------------------------------------------------------------------

_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _sanitize_name(name: str) -> str:
    """Make a Chroma-safe collection name: 3-63 chars, [a-zA-Z0-9_-], start/end alnum."""
    s = _SANITIZE_RE.sub("_", name).strip("_-")
    if not s or not s[0].isalnum():
        s = "c_" + s
    s = s[:63]
    if len(s) < 3:
        s = (s + "___")[:3]
    return s


def _collection_name(kind: str, user_name: str) -> str:
    """Build the full collection name: 'pdf_<sanitized>' or 'paste_<sanitized>'."""
    return f"{kind}_{_sanitize_name(user_name)}"


# ---------------------------------------------------------------------------
# Embedding model (loaded once, process-wide)
# ---------------------------------------------------------------------------

_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.retrieval.embedding_model)
    return _embedder


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="AEDE Backend", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _warm_embedder() -> None:
    """Pre-load the embedding model so the first /optimize call doesn't pay
    the ~30s cold-load cost and trip the Next.js dev proxy's 30s timeout."""
    _get_embedder()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


@app.get("/collections")
def collections_list() -> list[dict]:
    return list_aede_collections()


@app.delete("/collections/{name}")
def collections_delete(name: str) -> dict:
    deleted = delete_aede_collection(name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Collection not found: {name}")
    return {"deleted": name}


@app.post("/collections/pdf")
async def collections_add_pdf(
    name: str = Form(...),
    file: UploadFile = File(...),
) -> dict:
    """Add a PDF as a named collection."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    # Extract text per page, chunk
    reader = PdfReader(__import__("io").BytesIO(raw))
    all_chunks: list[tuple[str, str]] = []      # (id, text)
    for page_idx, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        for chunk_idx, chunk in enumerate(semantic_chunks(text)):
            cid = f"p{page_idx + 1}::c{chunk_idx}"
            all_chunks.append((cid, chunk))

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No extractable text in PDF.")

    coll_name = _collection_name("pdf", name)
    client = get_chroma_client()
    # Overwrite if it exists
    try:
        client.delete_collection(coll_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=coll_name,
        metadata={
            "hnsw:space": "cosine",
            "source_kind": "pdf",
            "filename": file.filename,
            "display_name": name,
        },
    )

    embedder = _get_embedder()
    texts = [t for _, t in all_chunks]
    embeddings = embedder.encode(texts).tolist()
    collection.add(
        ids=[cid for cid, _ in all_chunks],
        documents=texts,
        embeddings=embeddings,
    )

    return {
        "name": coll_name,
        "display_name": name,
        "type": "pdf",
        "items": len(all_chunks),
        "filename": file.filename,
    }


@app.post("/collections/paste")
async def collections_add_paste(
    name: str = Form(...),
    text: str = Form(...),
    kind: str = Form("chat"),                  # "chat" | "agent" — label only
) -> dict:
    """Add a text blob as a named paste collection."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty text.")
    if kind not in {"chat", "agent"}:
        raise HTTPException(status_code=400, detail="kind must be 'chat' or 'agent'.")

    chunks = semantic_chunks(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Text produced no chunks.")

    coll_name = _collection_name("paste", name)
    client = get_chroma_client()
    try:
        client.delete_collection(coll_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=coll_name,
        metadata={"hnsw:space": "cosine", "source_kind": kind, "display_name": name},
    )

    embedder = _get_embedder()
    embeddings = embedder.encode(chunks).tolist()
    ids = [f"c{i}" for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, embeddings=embeddings)

    return {
        "name": coll_name,
        "display_name": name,
        "type": "paste",
        "kind": kind,
        "items": len(chunks),
    }


# ---------------------------------------------------------------------------
# Optimize
# ---------------------------------------------------------------------------


class OptimizeRequest(BaseModel):
    collections: list[str]
    query: str


class OptimizeResponse(BaseModel):
    answer: str
    decision: str
    final_tokens: int
    raw_tokens: int
    saved_pct: float
    items_count: int
    collections_used: list[str]
    trace: list[str]
    coverage: float
    workflow_path: list[str]
    timings: list[dict] = []                  # [{node, model, elapsed_ms}]
    total_ms: int = 0


def _raw_gemini_tokens(query: str, documents: list[str]) -> int:
    """Send the same docs to Gemini verbatim and return total tokens."""
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY") or settings.models.gemini_api_key
    if not api_key:
        return 0
    docs_text = "\n\n- ".join(documents)
    contents = f"{query}. based on the following documents only: \n\n- {docs_text}"
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=settings.models.gemini_reasoner_model,
        contents=contents,
    )
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return 0
    total = (
        getattr(usage, "total_token_count", None)
        or getattr(usage, "total_tokens", None)
        or 0
    )
    if not total:
        in_t = getattr(usage, "prompt_token_count", 0) or 0
        out_t = getattr(usage, "candidates_token_count", 0) or 0
        total = in_t + out_t
    return int(total)


def _embed_query_for_retrieval(query: str) -> list[float]:
    """Embed a query string for the in-memory scratch collection the AEDE
    pipeline reads from. This is the one piece of plumbing that connects
    multi-collection query → single-collection AEDE pipeline."""
    embedder = _get_embedder()
    return embedder.encode([query])[0].tolist()


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest) -> OptimizeResponse:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query.")
    if not req.collections:
        raise HTTPException(status_code=400, detail="Select at least one collection.")

    # 1. Pull top-k from each selected collection, write the union into the
    #    per-request scratch collection. AEDE will read from there.
    per_k = settings.retrieval.initial_k
    pooled_docs = query_pool(req.collections, req.query, per_collection_k=per_k)
    if not pooled_docs:
        raise HTTPException(
            status_code=400,
            detail="Selected collections produced no documents. Are they empty?",
        )

    client = get_chroma_client()
    from aede.nodes.retrieval import REQUEST_COLLECTION
    try:
        client.delete_collection(REQUEST_COLLECTION)
    except Exception:
        pass
    scratch = client.get_or_create_collection(
        name=REQUEST_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # Embed the pooled docs once and write to scratch. AEDE queries scratch
    # with text; Chroma will embed the query itself (same model), so we just
    # need the documents in there.
    embedder = _get_embedder()
    embeddings = embedder.encode(pooled_docs).tolist()
    ids = [f"pooled_{i}" for i in range(len(pooled_docs))]
    scratch.add(ids=ids, documents=pooled_docs, embeddings=embeddings)

    # 2. Run AEDE
    initial = create_initial_state(req.query)
    graph = get_graph()
    final, timings, total_ms = run_with_timings(graph, initial)

    answer = final.get("answer", "") or ""
    decision = _decision_from_path(final.get("workflow_path", []))
    token_usage = final.get("token_usage", {}) or {}
    final_tokens = int(token_usage.get("total", 0) or 0)
    coverage = float(final.get("coverage", 0.0) or 0.0)

    # 3. Raw-Gemini baseline — same docs, no optimization
    try:
        raw_tokens = _raw_gemini_tokens(req.query, pooled_docs)
    except Exception:  # noqa: BLE001
        raw_tokens = 0

    saved_pct = (raw_tokens - final_tokens) / raw_tokens if raw_tokens else 0.0

    # 4. Drop scratch collection (pooled docs were already in the source
    #    collections; the scratch is just a working copy)
    try:
        client.delete_collection(REQUEST_COLLECTION)
    except Exception:
        pass

    return OptimizeResponse(
        answer=answer,
        decision=decision,
        final_tokens=final_tokens,
        raw_tokens=raw_tokens,
        saved_pct=saved_pct,
        items_count=len(pooled_docs),
        collections_used=req.collections,
        trace=[s for s in (final.get("workflow_path", []) or []) if s],
        coverage=coverage,
        workflow_path=final.get("workflow_path", []) or [],
        timings=timings,
        total_ms=total_ms,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decision_from_path(workflow_path: list[str]) -> str:
    last = next(
        (s for s in reversed(workflow_path or []) if isinstance(s, str) and s.startswith("compile(")),
        "",
    )
    if last.startswith("compile(") and last.endswith(")"):
        return last[len("compile(") : -1]
    return "unknown"
