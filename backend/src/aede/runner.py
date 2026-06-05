"""Streaming runner: yields step events while the LangGraph executes.

We don't instrument every node with timestamps — instead we use LangGraph's
`astream(stream_mode="updates")` and capture the wall-clock time around each
node transition. The state diff gives us the new workflow_path entry; we
infer which step just finished from the *last* entry in that path.

The runner is the only place that knows the SSE wire format. Consumers get an
async iterator of `dict` events; the FastAPI route serializes them.
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator

from langgraph.graph import StateGraph

from aede.models import model_for


def _last_step_name(workflow_path: list[str]) -> str | None:
    """Return the node name associated with the most recent workflow_path entry.

    LangGraph emits the full updated state for each node, so we compare against
    the previous path to find the newly-appended entry.
    """
    for entry in reversed(workflow_path or []):
        if not entry:
            continue
        # compiler uses "compile(<decision>)" — strip the parenthetical
        if entry.startswith("compile("):
            return "compile"
        if entry.startswith("retrieve_more"):
            return "retrieve_more"
        if entry == "focused_retriever":
            return "focused_retriever"
        if entry in {"extract", "analyze", "compress", "reason", "small_reasoner"}:
            return entry
        return entry
    return None


async def stream_run(
    graph: StateGraph,
    initial_state: dict[str, Any],
) -> AsyncIterator[dict[str, Any]]:
    """Run the compiled graph and yield step events.

    Yields events of two kinds:

      {"event": "step_started",  "node": "extract", "model": "Llama 8B", "t": 0.0}
      {"event": "step_finished", "node": "extract", "model": "Llama 8B",
       "elapsed_ms": 1423, "workflow_path": [...], "coverage": 0.5}
      {"event": "done", "final_state": {...}, "total_ms": 4721}
      {"event": "error", "message": "..."}
    """
    started = time.perf_counter()
    previous_path: list[str] = []
    node_started_at: float | None = None
    current_node: str | None = None
    # Merge diffs as we go so the final state we ship is the real one,
    # not a re-invocation of the whole graph.
    merged: dict[str, Any] = dict(initial_state)

    try:
        async for chunk in graph.astream(initial_state, stream_mode="updates"):
            # chunk is {node_name: partial_state_diff}
            for node_name, partial in chunk.items():
                now = time.perf_counter()
                workflow_path = (partial or {}).get("workflow_path") or previous_path
                # If a new node is starting, close out the previous one first.
                if current_node is not None and current_node != node_name:
                    elapsed_ms = int((now - (node_started_at or now)) * 1000)
                    yield {
                        "event": "step_finished",
                        "node": current_node,
                        "model": model_for(current_node),
                        "elapsed_ms": elapsed_ms,
                        "workflow_path": previous_path,
                    }
                    previous_path = workflow_path
                if current_node != node_name:
                    current_node = node_name
                    node_started_at = now
                    yield {
                        "event": "step_started",
                        "node": node_name,
                        "model": model_for(node_name),
                        "t": int((now - started) * 1000),
                    }
                # Accumulate the diff into the merged state. Keys in the
                # diff overwrite keys in the merged state — that matches
                # how LangGraph's StateReducer treats TypedDict updates.
                if partial:
                    merged.update(partial)
        # Close out the final node.
        if current_node is not None:
            now = time.perf_counter()
            elapsed_ms = int((now - (node_started_at or now)) * 1000)
            yield {
                "event": "step_finished",
                "node": current_node,
                "model": model_for(current_node),
                "elapsed_ms": elapsed_ms,
                "workflow_path": previous_path,
            }

        # No re-invocation — the merged state already has the final values.
        total_ms = int((time.perf_counter() - started) * 1000)
        yield {"event": "done", "final_state": merged, "total_ms": total_ms}
    except Exception as exc:  # noqa: BLE001
        yield {"event": "error", "message": str(exc)}
