"""FastAPI sidecar for the AEDE frontend.

Endpoints:
  GET  /health        — liveness probe
  GET  /stats         — global page/chunk/embedding counts
  POST /upload        — multipart PDF upload (single shared collection)
  POST /chat          — one-shot AEDE run (kept for back-compat)
  POST /chat/stream   — SSE stream: step_started, step_finished, done
  POST /raw-gemini    — baseline Gemini call for the metrics card
"""

from __future__ import annotations

import io
import json
import os
import re
from typing import Any, AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader

from aede.config import settings
from aede.graph import get_graph
from aede.runner import stream_run
from aede.state import create_initial_state
from aede.nodes.retrieval import add_documents, focused_retriever, get_or_create_vectorstore


# ---------------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------------

TARGET_CHUNK = 1000
OVERLAP = 200
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _semantic_chunks(text: str) -> list[str]:
    """Sentence-aware chunker: respects paragraph boundaries, ~1000 char target
    with 200 char overlap. Falls back to sentence splitting on long paragraphs.
    """
    text = text.strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buffer = ""

    def flush(buf: str) -> None:
        if buf.strip():
            chunks.append(buf.strip())

    for para in paragraphs:
        if len(para) > TARGET_CHUNK:
            sentences = SENTENCE_END.split(para)
            for sent in sentences:
                if len(buffer) + len(sent) + 1 > TARGET_CHUNK and buffer:
                    flush(buffer)
                    buffer = buffer[-OVERLAP:] + " " + sent
                else:
                    buffer = (buffer + " " + sent).strip()
        else:
            if len(buffer) + len(para) + 2 > TARGET_CHUNK and buffer:
                flush(buffer)
                buffer = buffer[-OVERLAP:] + "\n\n" + para
            else:
                buffer = (buffer + "\n\n" + para).strip()
    flush(buffer)
    return chunks


def _pdf_to_chunks(file_bytes: bytes) -> tuple[int, list[str]]:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = len(reader.pages)
    all_chunks: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        all_chunks.extend(_semantic_chunks(text))
    return pages, all_chunks


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    query: str


class RawGeminiRequest(BaseModel):
    query: str


class UploadResponse(BaseModel):
    pages: int
    chunks: int
    embeddings: int
    filename: str


class StatsResponse(BaseModel):
    pages: int
    chunks: int
    embeddings: int
    last_filename: str | None = None


# Out-of-band metadata (pages don't live in Chroma, so we track the most
# recent upload's page count alongside the running embedding total).
_LAST_UPLOAD: dict[str, int] = {"pages": 0, "chunks": 0, "filename": None}


def _shape_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer": state.get("answer", ""),
        "workflow_path": state.get("workflow_path", []),
        "route": _route_from_path(state.get("workflow_path", [])),
        "coverage": state.get("coverage", 0.0),
        "coverage_history": state.get("coverage_history", []),
        "token_usage": state.get("token_usage", {}),
        "current_top_k": state.get("current_top_k", 0),
        "required_reasoning": state.get("required_reasoning", "deep"),
        "max_retrieval_reached": state.get("max_retrieval_reached", False),
        "error": state.get("error"),
    }


def _route_from_path(workflow_path: list[str]) -> str:
    last = next(
        (s for s in reversed(workflow_path or []) if s.startswith("compile(")),
        "",
    )
    if last.startswith("compile(") and last.endswith(")"):
        return last[len("compile(") : -1]
    return "unknown"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="AEDE Backend", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    coll = get_or_create_vectorstore()
    return StatsResponse(
        pages=_LAST_UPLOAD["pages"],
        chunks=_LAST_UPLOAD["chunks"],
        embeddings=coll.count(),
        last_filename=_LAST_UPLOAD["filename"],
    )


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        pages, chunks = _pdf_to_chunks(raw)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {exc}") from exc

    if not chunks:
        raise HTTPException(status_code=400, detail="No extractable text in PDF.")

    ids = [f"{file.filename}::chunk::{i}" for i in range(len(chunks))]
    add_documents(chunks, ids=ids)

    _LAST_UPLOAD["pages"] = pages
    _LAST_UPLOAD["chunks"] = len(chunks)
    _LAST_UPLOAD["filename"] = file.filename

    coll = get_or_create_vectorstore()
    return UploadResponse(
        pages=pages,
        chunks=len(chunks),
        embeddings=coll.count(),
        filename=file.filename,
    )


# ---------------------------------------------------------------------------
# Chat (one-shot) + Chat (SSE stream)
# ---------------------------------------------------------------------------


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query.")
    initial = create_initial_state(req.query)
    graph = get_graph()
    final = graph.invoke(initial)
    return _shape_state(final)


async def _sse_format(event: str, data: dict[str, Any]) -> bytes:
    """Format one SSE event. `data` is a JSON-serializable dict."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n".encode()


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE stream of step events + a final 'done' event with the full state."""
    from fastapi.responses import StreamingResponse

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query.")

    initial = create_initial_state(req.query)
    graph = get_graph()

    async def event_source() -> AsyncIterator[bytes]:
        # Tiny "started" frame so the UI knows we're live.
        yield await _sse_format("started", {"query": req.query})
        try:
            async for ev in stream_run(graph, initial):
                kind = ev.get("event")
                if kind == "done":
                    final = ev.get("final_state") or {}
                    yield await _sse_format(
                        "done",
                        {
                            "state": _shape_state(final),
                            "total_ms": ev.get("total_ms", 0),
                        },
                    )
                elif kind == "error":
                    yield await _sse_format("error", {"message": ev.get("message", "")})
                else:
                    yield await _sse_format(kind, ev)
        except Exception as exc:  # noqa: BLE001
            yield await _sse_format("error", {"message": str(exc)})
        yield b"\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Raw-Gemini baseline
# ---------------------------------------------------------------------------


@app.post("/raw-gemini")
def raw_gemini(req: RawGeminiRequest) -> dict[str, Any]:
    """Baseline: ask Gemini 2.5 Flash directly with the same retrieved docs."""
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY") or settings.models.gemini_api_key
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set in backend/.env")

    initial = create_initial_state(req.query)
    retrieved = focused_retriever(initial)
    docs = retrieved.get("documents", [])
    docs_text = "\n\n- ".join(docs)

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=settings.models.gemini_reasoner_model,
        contents=f"{req.query}. based on the following documents only: \n\n- {docs_text}",
    )

    usage = getattr(response, "usage_metadata", None)
    total = 0
    if usage is not None:
        total = (
            getattr(usage, "total_token_count", None)
            or getattr(usage, "total_tokens", None)
            or 0
        )

    return {
        "tokens": int(total),
        "model": settings.models.gemini_reasoner_model,
        "retrieved_docs": len(docs),
    }
