"""Evidence Compressor node using Groq API."""

import json
import re

from aede.state import AEDEState
from aede.utils.groq_client import generate_with_groq


def evidence_compressor(state: AEDEState) -> AEDEState:
    """
    Compress extracted facts to reduce redundancy while preserving unique information.
    Uses Groq API. Target: 10x reduction in evidence count.
    Token-optimized: limits facts, truncates claims/quotes, caps total content.
    """
    query = state["query"]
    facts = state.get("facts", [])

    if not facts:
        return {**state, "compressed_evidence": [], "workflow_path": state.get("workflow_path", []) + ["compress"]}

    # Token optimization: limit facts and truncate
    max_facts = 40
    facts = facts[:max_facts]

    facts_text_parts = []
    total_chars = 0
    max_chars = 6000

    for f in facts:
        fact_str = f"- Claim: {f['claim'][:150]}\n  Quote: {f['quote'][:100] if f['quote'] else ''}\n"
        if total_chars + len(fact_str) > max_chars:
            break
        facts_text_parts.append(fact_str)
        total_chars += len(fact_str)

    facts_text = "".join(facts_text_parts)

    system_prompt = """Merge redundant claims, preserve unique info. Return ONLY valid JSON:
{"compressed_evidence": ["evidence 1", "evidence 2"]}"""

    user_prompt = f"Query: {query}\n\nFacts:\n{facts_text}\n\n10x reduction."

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
        current_usage["compressor_input"] = current_usage.get("compressor_input", 0) + usage.get("prompt_tokens", 0)
        current_usage["compressor_output"] = current_usage.get("compressor_output", 0) + usage.get("completion_tokens", 0)

    # Parse JSON response
    compressed_evidence = []
    try:
        parsed = json.loads(response_text)
        if "compressed_evidence" in parsed:
            compressed_evidence = [str(e) for e in parsed["compressed_evidence"]]
    except json.JSONDecodeError:
        json_match = re.search(r"\{[^}]*\"compressed_evidence\"\s*:\s*(\[[^\]]*)\}", response_text, re.DOTALL)
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
        "error": result.get("error") if "error" in result else None,
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
