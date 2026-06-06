"""Evidence Extractor node using Groq API."""

import json
import re

from aede.state import AEDEState, Fact
from aede.utils.groq_client import generate_with_groq


def extractor(state: AEDEState) -> AEDEState:
    """
    Extract factual claims from retrieved documents using Groq API.
    Optimized for token efficiency: limited docs, capped chunk size, total budget.
    """
    query = state["query"]
    documents = state.get("documents", [])

    if not documents:
        return {**state, "facts": [], "workflow_path": state.get("workflow_path", []) + ["extract"]}

    # Token optimization: limit docs, chunk size, and total content
    max_docs = 10
    max_chunk_chars = 1500
    max_total_chars = 6000                              # was 12000 — cuts input ~50%

    documents = documents[:max_docs]

    # Build prompt with content limits
    user_prompt_parts = []
    total_chars = 0
    for i, doc in enumerate(documents):
        # Truncate each chunk
        truncated = doc[:max_chunk_chars] + ("..." if len(doc) > max_chunk_chars else "")
        doc_with_header = f"\n[Document {i}]:\n{truncated}\n"

        # Check if adding this would exceed budget
        if total_chars + len(doc_with_header) > max_total_chars:
            break

        user_prompt_parts.append(doc_with_header)
        total_chars += len(doc_with_header)

    user_prompt = f"Query: {query}\n\nDocuments:" + "".join(user_prompt_parts)

    system_prompt = """Extract factual claims from documents relevant to the query.
Respond ONLY with valid JSON: {"facts": [{"claim": "...", "quote": "...", "chunk_id": 0}]}
If no relevant facts, respond: {"facts": []}"""

    # Call Groq API
    result = generate_with_groq(
        prompt=user_prompt,
        system_prompt=system_prompt,
        json_mode=True,
    )

    response_text = result.get("text", "")
    usage = result.get("usage", {})

    # Update token tracking
    current_usage = state.get("token_usage", {})
    if usage:
        current_usage["extractor_input"] = current_usage.get("extractor_input", 0) + usage.get("prompt_tokens", 0)
        current_usage["extractor_output"] = current_usage.get("extractor_output", 0) + usage.get("completion_tokens", 0)

    # Parse JSON response
    facts: list[Fact] = []
    try:
        # Try direct parse first
        parsed = json.loads(response_text)
        if "facts" in parsed:
            facts = [Fact(claim=str(f.get("claim", "")), quote=str(f.get("quote", "")), chunk_id=int(f.get("chunk_id", 0))) for f in parsed["facts"]]
    except json.JSONDecodeError:
        # Try markdown code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL) or re.search(r"(\{[^}]*\"facts\"\s*:\s*\[[^\]]*\])", response_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if "facts" in parsed:
                    facts = [Fact(claim=str(f.get("claim", "")), quote=str(f.get("quote", "")), chunk_id=int(f.get("chunk_id", 0))) for f in parsed["facts"]]
            except json.JSONDecodeError:
                pass

    return {
        **state,
        "facts": facts,
        "token_usage": current_usage,
        "workflow_path": state.get("workflow_path", []) + ["extract"],
        "error": result.get("error") if "error" in result else None,
    }
