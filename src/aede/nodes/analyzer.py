"""Evidence Quality Analyzer node using Gemini API."""

import json
import os
import re

from aede.state import AEDEState


def analyzer(state: AEDEState) -> AEDEState:
    """
    Analyze the quality of extracted evidence using Gemini API.
    """
    query = state["query"]
    query_core_concepts = state.get("query_core_concepts", [])
    facts = state.get("facts", [])

    if not facts:
        return {**state, "answered_parts": [], "missing_parts": [], "missing_parts_core": [], "coverage": 0.0, "redundancy": 0.0, "confidence": 0.0, "workflow_path": state.get("workflow_path", []) + ["analyze"]}

    # Build prompt
    system_prompt = """You are an evidence quality analyst. Evaluate if extracted facts answer the query.

Respond ONLY with valid JSON:
{
  "answered_parts": ["part1"],
  "missing_parts": ["missing part"],
  "coverage": 0.85,
  "redundancy": 0.2,
  "confidence": 0.8
}"""

    facts_text = "\n".join([f"- Claim: {f['claim']}\n  Quote: {f['quote']}\n  Source chunk: {f['chunk_id']}" for f in facts])
    user_prompt = f"""Query: {query}

Extracted Facts:
{facts_text}

Analyze the evidence quality."""

    # Get API key
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return {**state, "error": "GEMINI_API_KEY not set", "workflow_path": state.get("workflow_path", []) + ["analyze"]}

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
        return {**state, "error": f"Analyzer API error: {str(e)}", "workflow_path": state.get("workflow_path", []) + ["analyze"]}

    # Update token tracking
    current_usage = state.get("token_usage", {})
    if usage_metadata:
        current_usage["analyzer_input"] = current_usage.get("analyzer_input", 0) + (usage_metadata.prompt_token_count or 0)
        current_usage["analyzer_output"] = current_usage.get("analyzer_output", 0) + (usage_metadata.candidates_token_count or 0)

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
        "error": None,
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