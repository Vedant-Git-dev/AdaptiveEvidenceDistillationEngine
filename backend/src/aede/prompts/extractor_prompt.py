"""Extractor prompt for Gemma 4 - Atomic fact extraction from documents."""

from typing import List


def get_extractor_prompt(documents: List[str], query: str) -> str:
    """
    Generate prompt for Gemma 4 to extract atomic facts from documents.

    Args:
        documents: List of document chunks
        query: The user's query/question

    Returns:
        Formatted prompt string
    """
    return f"""<start_of_turn>
You are a precise fact extraction system. Given a query and documents, extract discrete atomic facts relevant to answering the query.

QUERY: {query}

DOCUMENTS:
{'-' * 60}

{chr(10).join(f"[Chunk {i}] {doc}" for i, doc in enumerate(documents))}

{'-' * 60}

TASK:
Extract ATOMIC FACTS only - each fact must be:
- A single verifiable claim (true or false)
- Directly relevant to answering the query
- Self-contained (no pronouns or references to other facts)

For each fact, also provide:
- The exact quote text from the source
- The chunk_id it came from

Respond ONLY with valid JSON in this exact format:
[
  {{"claim": "fact statement", "quote": "exact text from document", "chunk_id": 0}},
  ...
]

Rules:
- Extract 50-150 facts maximum
- Each fact should be 1-2 sentences
- Include facts that partially answer the query
- Skip duplicate or near-duplicate facts
- Include negative/unverified claims if relevant
- Do not add commentary or explanations
- Only output the JSON array
<end_of_turn>"""