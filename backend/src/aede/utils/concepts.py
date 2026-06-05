"""Core concept extraction and manipulation utilities.

These functions wrap concept_extractor.py functionality and provide
standalone utilities for concept manipulation.
"""

from typing import List, Set, Optional, Dict, Any
from collections import Counter


def overlap_score(concepts1: List[str], concepts2: List[str]) -> float:
    """Calculate Jaccard similarity between two concept lists.

    Jaccard similarity = |A ∩ B| / |A ∪ B|

    Args:
        concepts1: First list of concepts
        concepts2: Second list of concepts

    Returns:
        Jaccard similarity score between 0.0 and 1.0
    """
    if not concepts1 and not concepts2:
        return 1.0
    if not concepts1 or not concepts2:
        return 0.0

    set1 = set(concepts1)
    set2 = set(concepts2)

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0.0

    return intersection / union


def extract_missing_core(
    answered_concepts: List[str],
    query_concepts: List[str],
) -> List[str]:
    """Extract concepts from query that are not covered by answered concepts.

    Args:
        answered_concepts: Concepts that were answered/covered
        query_concepts: All concepts in the original query

    Returns:
        List of concepts from query_concepts not in answered_concepts
    """
    answered_set = set(c.lower() for c in answered_concepts)
    return [c for c in query_concepts if c.lower() not in answered_set]


def extract_missing_core_fuzzy(
    answered_concepts: List[str],
    query_concepts: List[str],
    similarity_threshold: float = 0.8,
) -> List[str]:
    """Extract missing concepts using fuzzy matching.

    Args:
        answered_concepts: Concepts that were answered/covered
        query_concepts: All concepts in the original query
        similarity_threshold: Minimum similarity to consider as match

    Returns:
        List of concepts from query_concepts not covered by answered_concepts
    """
    answered_set = set(c.lower() for c in answered_concepts)
    missing = []

    for query in query_concepts:
        query_lower = query.lower()
        matched = False

        for answered in answered_set:
            if query_lower == answered:
                matched = True
                break
            # Check if one contains the other
            if query_lower in answered or answered in query_lower:
                matched = True
                break

        if not matched:
            missing.append(query)

    return missing


def concept_precision_recall(
    predicted_concepts: List[str],
    true_concepts: List[str],
) -> Dict[str, float]:
    """Calculate precision and recall for concept extraction.

    Args:
        predicted_concepts: Concepts predicted by the model
        true_concepts: Ground truth concepts

    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    pred_set = set(predicted_concepts)
    true_set = set(true_concepts)

    true_positives = len(pred_set & true_set)
    false_positives = len(pred_set - true_set)
    false_negatives = len(true_set - pred_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def dedupe_concepts(concepts: List[str]) -> List[str]:
    """Remove duplicate concepts (case-insensitive).

    Args:
        concepts: List of concept strings

    Returns:
        Deduplicated list preserving original order
    """
    seen: Set[str] = set()
    result = []

    for concept in concepts:
        lower = concept.lower()
        if lower not in seen:
            seen.add(lower)
            result.append(concept)

    return result


def group_concepts_by_theme(concepts: List[str]) -> Dict[str, List[str]]:
    """Group concepts by common themes/topics.

    Args:
        concepts: List of concept strings

    Returns:
        Dictionary mapping themes to lists of concepts
    """
    # Simple keyword-based theme detection
    theme_keywords = {
        "model": ["model", "llm", "ai", "gpt", "gemini", "gemma", "neural", "training"],
        "retrieval": ["retrieval", "search", "vector", "embed", "chroma", "rag"],
        "data": ["data", "token", "input", "output", "document", "text"],
        "evaluation": ["eval", "benchmark", "quality", "score", "accuracy", "metrics"],
        "configuration": ["config", "setting", "parameter", "hyperparameter", "threshold"],
    }

    theme_map: Dict[str, List[str]] = {}
    ungrouped: List[str] = []

    for concept in concepts:
        concept_lower = concept.lower()
        grouped = False

        for theme, keywords in theme_keywords.items():
            if any(kw in concept_lower for kw in keywords):
                if theme not in theme_map:
                    theme_map[theme] = []
                theme_map[theme].append(concept)
                grouped = True
                break

        if not grouped:
            ungrouped.append(concept)

    if ungrouped:
        theme_map["other"] = ungrouped

    return theme_map


def normalize_concepts(concepts: List[str]) -> List[str]:
    """Normalize concepts for comparison.

    Args:
        concepts: List of concept strings

    Returns:
        Normalized list of concepts
    """
    result = []
    for concept in concepts:
        # Lowercase and strip
        normalized = concept.lower().strip()
        # Remove redundant whitespace
        normalized = " ".join(normalized.split())
        result.append(normalized)
    return dedupe_concepts(result)