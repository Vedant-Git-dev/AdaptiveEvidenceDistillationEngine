"""Utility: Extract core concepts from query."""

from aede.state import AEDEState
from aede.config import settings


# Simple keyword extraction - no LLM needed
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "but", "if", "or", "because", "until", "while", "about", "against",
    "what", "which", "who", "whom", "this", "that", "these", "those", "it", "its",
}


def extract_core_concepts(state: AEDEState) -> AEDEState:
    """
    Extract core concepts from query.

    Simple keyword extraction approach:
    - Split query into words
    - Remove stop words
    - Keep meaningful nouns/verbs

    This is a fast, no-LLM approach. For better results,
    could use Gemma 4 for semantic concept extraction.

    Returns:
        Updated state with query_core_concepts
    """
    query = state["query"].lower()

    # Tokenize and filter
    words = query.split()
    concepts = []

    for word in words:
        # Remove punctuation
        clean = "".join(c for c in word if c.isalnum())
        # Skip stop words and very short words
        if clean and clean not in STOP_WORDS and len(clean) > 2:
            concepts.append(clean)

    # Deduplicate while preserving order
    seen = set()
    unique_concepts = []
    for c in concepts:
        if c not in seen:
            seen.add(c)
            unique_concepts.append(c)

    return {
        "query_core_concepts": unique_concepts,
    }


def extract_concepts_advanced(query: str) -> list[str]:
    """
    Advanced concept extraction using simple NLP heuristics.

    Groups related terms and prioritizes important concepts.
    """
    # Extract n-grams (bigrams and trigrams)
    words = query.lower().split()
    ngrams = []

    # Clean words
    clean_words = []
    for w in words:
        clean = "".join(c for c in w if c.isalnum())
        if clean and clean not in STOP_WORDS:
            clean_words.append(clean)

    # Extract bigrams
    for i in range(len(clean_words) - 1):
        ngrams.append(f"{clean_words[i]}_{clean_words[i+1]}")

    # Combine single words and bigrams
    all_concepts = clean_words + ngrams

    # Deduplicate
    seen = set()
    unique = []
    for c in all_concepts:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return unique