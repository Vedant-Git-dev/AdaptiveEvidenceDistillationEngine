"""Evidence Quality Analyzer node using Groq API."""

import json
import re

from aede.state import AEDEState
from aede.utils.groq_client import generate_with_groq


def analyzer(state: AEDEState) -> AEDEState:
    """
    Analyze the quality of extracted evidence using Groq API.
    """
    query = state["query"]
    query_core_concepts = state.get("query_core_concepts", [])
    facts = state.get("facts", [])

    if not facts:
        return {**state, "answered_parts": [], "missing_parts": [], "missing_parts_core": [], "coverage": 0.0, "redundancy": 0.0, "confidence": 0.0, "workflow_path": state.get("workflow_path", []) + ["analyze"]}

    # Build prompt with token limits
    max_facts = 50
    facts = facts[:max_facts]

    facts_text_parts = []
    total_chars = 0
    max_chars = 8000

    for f in facts:
        fact_str = f"Claim: {f['claim'][:200]}\n  Quote: {f['quote'][:150] if f['quote'] else ''}\n"
        if total_chars + len(fact_str) > max_chars:
            break
        facts_text_parts.append(fact_str)
        total_chars += len(fact_str)

    facts_text = "".join(facts_text_parts)

    system_prompt = """Evaluate if facts answer the query. JSON only:
{"answered_parts": [], "missing_parts": [], "coverage": 0.0, "redundancy": 0.0, "confidence": 0.0}"""

    user_prompt = f"Query: {query}\n\nFacts:\n{facts_text}\n\nAnalyze quality."

    # Call Groq API
    result = generate_with_groq(
        prompt=user_prompt,
        system_prompt=system_prompt,
    )

    response_text = result.get("text", "")
    usage = result.get("usage", {})

    # Update token tracking
    current_usage = state.get("token_usage", {})
    if usage:
        current_usage["analyzer_input"] = current_usage.get("analyzer_input", 0) + usage.get("prompt_tokens", 0)
        current_usage["analyzer_output"] = current_usage.get("analyzer_output", 0) + usage.get("completion_tokens", 0)

    # Parse response
    answered_parts = []
    missing_parts = []
    coverage = 0.0
    redundancy = 0.0
    confidence = 0.0

    try:
        parsed = json.loads(response_text)
        answered_parts = [str(p) for p in parsed.get("answered_parts", [])]
        missing_parts = [str(p) for p in parsed.get("missing_parts", [])]
        coverage = float(parsed.get("coverage", 0.0))
        redundancy = float(parsed.get("redundancy", 0.0))
        confidence = float(parsed.get("confidence", 0.0))
    except (json.JSONDecodeError, ValueError):
        json_match = re.search(r"\{[^}]+\}", response_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                answered_parts = [str(p) for p in parsed.get("answered_parts", [])]
                missing_parts = [str(p) for p in parsed.get("missing_parts", [])]
                coverage = float(parsed.get("coverage", 0.0))
                redundancy = float(parsed.get("redundancy", 0.0))
                confidence = float(parsed.get("confidence", 0.0))
            except (json.JSONDecodeError, ValueError):
                pass

    # Calculate missing_parts_core
    missing_parts_core = _calculate_missing_parts_core(missing_parts, query_core_concepts, answered_parts)

    return {
        **state,
        "answered_parts": answered_parts,
        "missing_parts": missing_parts,
        "missing_parts_core": missing_parts_core,
        "coverage": coverage,
        "redundancy": redundancy,
        "confidence": confidence,
        "token_usage": current_usage,
        "workflow_path": state.get("workflow_path", []) + ["analyze"],
        "error": result.get("error") if "error" in result else None,
    }


def _calculate_missing_parts_core(missing_parts: list[str], query_core_concepts: list[str], answered_parts: list[str]) -> list[str]:
    """Calculate missing parts that intersect with core concepts."""
    if not missing_parts or not query_core_concepts:
        return []

    answered_keywords = set()
    core_lower = [c.lower() for c in query_core_concepts]

    for part in answered_parts:
        part_lower = part.lower()
        for concept in core_lower:
            if concept in part_lower:
                answered_keywords.add(concept)

    missing_core = []
    for part in missing_parts:
        part_lower = part.lower()
        for concept in core_lower:
            if concept in part_lower and concept not in answered_keywords:
                missing_core.append(part)
                break

    return missing_core
