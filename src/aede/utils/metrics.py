"""Evaluation metrics for AEDE."""

from typing import List, Optional
from pathlib import Path
import json


def compression_ratio(input_count: int, output_count: int) -> float:
    """Calculate compression ratio.

    Args:
        input_count: Number of input items (tokens, facts, etc.)
        output_count: Number of output items

    Returns:
        Compression ratio (input/output), or 0 if output is 0
    """
    if output_count == 0:
        return 0.0
    return input_count / output_count


def token_reduction_percent(aede_tokens: int, baseline_tokens: int) -> float:
    """Calculate token reduction percentage vs baseline.

    Args:
        aede_tokens: Tokens used by AEDE system
        baseline_tokens: Tokens used by baseline system

    Returns:
        Percentage reduction (0-100), negative if AEDE uses more tokens
    """
    if baseline_tokens == 0:
        return 0.0
    return ((baseline_tokens - aede_tokens) / baseline_tokens) * 100


def workflow_efficiency(workflow_path: str) -> int:
    """Count retrieval loops in a workflow trace.

    Counts how many times the workflow went back to retrieve more context.

    Args:
        workflow_path: Path to workflow trace file or JSON string

    Returns:
        Number of retrieval loops performed
    """
    if isinstance(workflow_path, (str, Path)):
        path = Path(workflow_path)
        if path.exists() and path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
        elif path.exists():
            raise ValueError(f"Workflow file must be JSON, got: {path.suffix}")
        else:
            # Try as JSON string
            try:
                data = json.loads(str(workflow_path))
            except json.JSONDecodeError:
                raise ValueError("workflow_path must be a valid JSON file path or JSON string")
    else:
        data = workflow_path

    # Count retrieval steps
    loops = 0

    def count_retrievals(obj):
        nonlocal loops
        if isinstance(obj, dict):
            # Count explicit retrieval steps
            if obj.get("action") == "retrieve" or obj.get("type") == "retrieval":
                loops += 1
            # Check for iteration/loop markers
            if obj.get("iteration") is not None:
                loops += 1
            # Recurse into nested structures
            for value in obj.values():
                count_retrievals(value)
        elif isinstance(obj, list):
            for item in obj:
                count_retrievals(item)

    count_retrievals(data)
    return loops


def calculate_coverage(
    answered_concepts: List[str],
    total_concepts: List[str],
) -> float:
    """Calculate concept coverage percentage.

    Args:
        answered_concepts: Concepts that were successfully answered
        total_concepts: All concepts that should have been answered

    Returns:
        Coverage percentage (0-100)
    """
    if not total_concepts:
        return 100.0

    answered_set = set(c.lower() for c in answered_concepts)
    covered = sum(1 for c in total_concepts if c.lower() in answered_set)
    return (covered / len(total_concepts)) * 100


def calculate_redundancy(items: List[str]) -> float:
    """Calculate redundancy score for a list of items.

    Uses Jaccard similarity between consecutive items to detect repetition.

    Args:
        items: List of items to check for redundancy

    Returns:
        Redundancy score (0-1), higher means more redundant
    """
    if len(items) < 2:
        return 0.0

    total_similarity = 0.0
    comparisons = 0

    for i in range(len(items) - 1):
        set1 = set(items[i].lower().split())
        set2 = set(items[i + 1].lower().split())

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union > 0:
            total_similarity += intersection / union
            comparisons += 1

    return total_similarity / comparisons if comparisons > 0 else 0.0


def mean_reciprocal_rank( relevance_lists: List[List[bool]]) -> float:
    """Calculate Mean Reciprocal Rank (MRR).

    Args:
        relevance_lists: List of lists, each containing boolean relevance for rankings

    Returns:
        MRR score
    """
    if not relevance_lists:
        return 0.0

    reciprocal_ranks = []

    for relevance in relevance_lists:
        for i, is_relevant in enumerate(relevance, 1):
            if is_relevant:
                reciprocal_ranks.append(1.0 / i)
                break
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def precision_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int,
) -> float:
    """Calculate precision at k.

    Args:
        retrieved: List of retrieved items
        relevant: List of relevant items
        k: Cutoff position

    Returns:
        Precision at k
    """
    if k <= 0:
        return 0.0

    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0

    relevant_set = set(r.lower() for r in relevant)
    hits = sum(1 for item in retrieved_k if item.lower() in relevant_set)

    return hits / k


def recall_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int,
) -> float:
    """Calculate recall at k.

    Args:
        retrieved: List of retrieved items
        relevant: List of relevant items
        k: Cutoff position

    Returns:
        Recall at k
    """
    if not relevant:
        return 1.0 if not retrieved else 0.0

    retrieved_k = retrieved[:k]
    relevant_set = set(r.lower() for r in relevant)
    hits = sum(1 for item in retrieved_k if item.lower() in relevant_set)

    return hits / len(relevant)