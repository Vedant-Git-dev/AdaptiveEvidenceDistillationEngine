"""Small-model Reasoner node (Groq).

Used when the analyzer reports that the question is easy:
  - required_reasoning == "none"  -> skip compressor, answer from raw facts
  - required_reasoning == "light" -> compressor already ran, answer from
                                     compressed_evidence

Uses the configured Groq pipeline model (the same small model used for
extractor/analyzer/compressor). Keeps cost and latency low for the easy path.
"""

from aede.state import AEDEState
from aede.utils.groq_client import generate_with_groq


# Token caps - small model gets a small budget
_MAX_EVIDENCE_ITEMS = 15
_MAX_CHARS_PER_ITEM = 500
_MAX_OUTPUT_TOKENS = 600


def small_reasoner(state: AEDEState) -> AEDEState:
    """
    Generate the final answer using the small (Groq) model.

    Evidence source is chosen by what the compiler put in state:
      - If compressed_evidence is non-empty (compress ran), use it.
      - Else fall back to top raw facts.
    """
    query = state["query"]
    required_reasoning = state.get("required_reasoning", "none")

    evidence: list[str] = []
    compressed = state.get("compressed_evidence", [])
    if compressed:
        evidence = list(compressed)
    else:
        facts = state.get("facts", []) or []
        evidence = [
            f"{f['claim']} - \"{f['quote']}\"" if f.get("quote") else f["claim"]
            for f in facts[:_MAX_EVIDENCE_ITEMS]
        ]

    if not evidence:
        return {
            **state,
            "answer": "Insufficient evidence to answer the query.",
            "workflow_path": state.get("workflow_path", []) + ["small_reasoner"],
        }

    # Token optimization: cap items and per-item length
    evidence = evidence[:_MAX_EVIDENCE_ITEMS]
    evidence = [
        e[:_MAX_CHARS_PER_ITEM] + ("..." if len(e) > _MAX_CHARS_PER_ITEM else "")
        for e in evidence
    ]

    evidence_text = "\n".join(f"- {e}" for e in evidence)

    system_prompt = (
        "You are a helpful assistant answering questions based on the provided evidence.\n"
        "Guidelines:\n"
        "- Answer concisely (target 150-400 words)\n"
        "- Use ONLY the evidence provided; if it is insufficient, say so plainly\n"
        "- Prefer direct, factual phrasing - the evidence has already been judged sufficient\n"
        f"- Reasoning depth required: {required_reasoning}\n"
    )

    user_prompt = f"Query: {query}\n\nEvidence:\n{evidence_text}\n\nProvide a clear, evidence-based answer."

    result = generate_with_groq(
        prompt=user_prompt,
        system_prompt=system_prompt,
        max_tokens=_MAX_OUTPUT_TOKENS,
        json_mode=False,                                  # free-form answer, not JSON
    )

    text = result.get("text", "")
    usage = result.get("usage", {})

    if not text:
        # API failure / no key - degrade gracefully so the user still gets an answer
        text = _simple_answer(query, evidence)

    current_usage = state.get("token_usage", {})
    if usage:
        in_t = usage.get("prompt_tokens", 0)
        out_t = usage.get("completion_tokens", 0)
        current_usage["small_reasoner_input"] = current_usage.get("small_reasoner_input", 0) + in_t
        current_usage["small_reasoner_output"] = current_usage.get("small_reasoner_output", 0) + out_t
        # `total` represents the FINAL model's bill (Llama in this branch).
        current_usage["total"] = in_t + out_t

    return {
        **state,
        "answer": text,
        "token_usage": current_usage,
        "workflow_path": state.get("workflow_path", []) + ["small_reasoner"],
        "error": result.get("error") if "error" in result else None,
    }


def _simple_answer(query: str, evidence: list[str]) -> str:
    """Fallback when Groq is unavailable."""
    if not evidence:
        return "No evidence available to answer the query."
    evidence_text = "\n\n".join(f"- {e}" for e in evidence[:10])
    return (
        f"Based on the available evidence, here is an answer to: {query}\n\n"
        f"{evidence_text}\n\n"
        "Note: This is an automated summary."
    )
