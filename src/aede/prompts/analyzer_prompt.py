"""Analyzer prompt for quality analysis of extracted facts."""

from typing import List, Dict, Any


def get_analyzer_prompt(
    facts: List[Dict[str, Any]],
    query: str,
    query_core_concepts: List[str]
) -> str:
    """
    Generate prompt for quality analysis of extracted facts.

    Args:
        facts: List of extracted facts with claim, quote, chunk_id
        query: The original user query
        query_core_concepts: Core concepts from the query

    Returns:
        Formatted prompt string
    """
    facts_json = "\n".join(
        f"  {{idx: {i}, claim: {f['claim']}, chunk: {f['chunk_id']}}}"
        for i, f in enumerate(facts)
    )

    return f"""<start_of_turn>
Analyze extracted facts for quality and completeness against the query.

QUERY: {query}
CORE CONCEPTS: {', '.join(query_core_concepts)}

EXTRACTED FACTS:
{facts_json}

{'-' * 60}

Analyze and return ONLY a JSON object with:
{{
  "answered_parts": ["aspect1", "aspect2"],
  "missing_parts": ["uncovered aspect1", "uncovered aspect2"],
  "coverage": 0.0-1.0,
  "redundancy": 0.0-1.0,
  "confidence": 0.0-1.0,
  "analysis_notes": "brief reasoning"
}}

Scoring criteria:
- coverage: Fraction of query aspects addressed by facts
- redundancy: How many facts overlap/duplicate each other (0=none, 1=all redundant)
- confidence: Overall reliability of extracted facts
- Include only concepts from CORE CONCEPTS in answered_parts/missing_parts
<end_of_turn>"""