"""Compressor prompt for evidence deduplication and reduction."""

from typing import List, Dict, Any


def get_compressor_prompt(facts: List[Dict[str, Any]]) -> str:
    """
    Generate prompt for compressing/deduplicating facts.

    Args:
        facts: List of extracted facts to compress

    Returns:
        Formatted prompt string
    """
    facts_json = "\n".join(
        f"  [{i}] {{claim: {f['claim']}, quote: {f['quote'][:100]}..., chunk: {f['chunk_id']}}}"
        for i, f in enumerate(facts)
    )

    return f"""<start_of_turn>
Compress and deduplicate facts to essential unique evidence. Target: 10x reduction.

INPUT FACTS ({len(facts)} total):
{facts_json}

{'-' * 60}

TASK:
1. Remove duplicate and near-duplicate facts (keep the most precise)
2. Merge overlapping facts into single consolidated facts
3. Keep only facts that add unique information value
4. Preserve the best quote for each fact

Return ONLY a JSON array of unique, non-redundant facts:
[
  {{"claim": "consolidated claim", "quote": "best source quote", "chunk_id": N, "source_indices": [orig_idx1, idx2]}},
  ...
]

Target output: {max(5, len(facts) // 10)} or fewer facts
- Aim for maximum 10x compression
- Each output fact should be more general than its inputs
- Do not lose key information or specific data points
- Merge quotes from multiple sources when they support the same claim
<end_of_turn>"""