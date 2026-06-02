"""Evidence Compressor node using Gemini API."""

import json
import os
import re

from aede.state import AEDEState


def evidence_compressor(state: AEDEState) -> AEDEState:
    """
    Compress extracted facts to reduce redundancy while preserving unique information.
    Uses Gemini API. Target: 10x reduction in evidence count.
    """
    query = state["query"]
    facts = state.get("facts", [])

    if not facts:
        return {**state, "compressed_evidence": [], "workflow_path": state.get("workflow_path", []) + ["compress"]}

    facts_text = "\n".join([f"- Claim: {f['claim']}\n  Quote: {f['quote']}\n  Source chunk: {f['chunk_id']}" for f in facts])

    system_prompt = """You are an evidence compressor. Merge redundant claims and preserve unique information.

Target: 10x reduction. Return ONLY valid JSON:
{"compressed_evidence": ["unique evidence 1", "unique evidence 2"]}"""

    user_prompt = f"Query: {query}\n\nFacts:\n{facts_text}\n\nMerge redundant claims keeping unique information."

    # Get API key
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return {**state, "compressed_evidence": _simple_compress(facts, query), "workflow_path": state.get("workflow_path", []) + ["compress"]}

    # Use official Google Gemini client
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemma-4-31b-it",
            contents=f"{system_prompt}\n\n{user_prompt}",
        )

        response_text = response.text
        usage_metadata = response.usage_metadata

    except Exception as e:
        return {**state, "compressed_evidence": _simple_compress(facts, query), "error": str(e), "workflow_path": state.get("workflow_path", []) + ["compress"]}

    # Update token tracking
    current_usage = state.get("token_usage", {})
    if usage_metadata:
        current_usage["compressor_input"] = current_usage.get("compressor_input", 0) + (usage_metadata.prompt_token_count or 0)
        current_usage["compressor_output"] = current_usage.get("compressor_output", 0) + (usage_metadata.candidates_token_count or 0)

    # Parse JSON response
    compressed_evidence = []
    try:
        parsed = json.loads(response_text)
        if "compressed_evidence" in parsed:
            compressed_evidence = [str(e) for e in parsed["compressed_evidence"]]
    except json.JSONDecodeError:
        json_match = re.search(r"\{[^}]*\"compressed_evidence\"\s*:\s*(\[[^\]]*\])", response_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                compressed_evidence = [str(e) for e in parsed.get("compressed_evidence", [])]
            except json.JSONDecodeError:
                pass

    if not compressed_evidence:
        compressed_evidence = _simple_compress(facts, query)

    return {
        **state,
        "compressed_evidence": compressed_evidence,
        "token_usage": current_usage,
        "workflow_path": state.get("workflow_path", []) + ["compress"],
        "error": None,
    }


def _simple_compress(facts: list, query: str) -> list[str]:
    """Simple rule-based compression."""
    seen = set()
    unique = []
    for fact in facts:
        key = fact["claim"].lower()[:50]
        if key not in seen:
            seen.add(key)
            unique.append(f"{fact['claim']} - \"{fact['quote']}\"")
    target = max(1, len(facts) // 10)
    return unique[:target] if len(unique) > target else unique