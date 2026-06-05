"""Prompts module for AEDE pipeline."""

from aede.prompts.compressor_prompt import get_compressor_prompt


# Compressor prompts
COMPRESSOR_SYSTEM_PROMPT = """You are an evidence compression expert. Your task is to reduce redundant facts to their essential unique information while preserving key data points and supporting quotes. Target: 10x reduction in evidence count."""

COMPRESSOR_USER_TEMPLATE = """
Query: {query}

Extracted Facts:
{facts_list}

TASK:
1. Remove duplicate and near-duplicate facts (keep the most precise)
2. Merge overlapping facts into single consolidated facts
3. Keep only facts that add unique information value
4. Preserve the best quote for each fact

Return ONLY a JSON array of unique evidence strings:
["evidence 1", "evidence 2", ...]
Target: {target} or fewer items
"""


# Reasoner prompts
REASONER_SYSTEM_PROMPT = """You are a precise information synthesis assistant. Based strictly on the provided evidence, generate a clear, accurate, and concise answer to the query. Cite evidence inline where applicable. If the evidence is insufficient, state that clearly."""

REASONER_USER_TEMPLATE = """
Query: {query}

Compressed Evidence:
{compressed_evidence}

Generate a concise, evidence-based answer targeting 300-800 tokens.
"""


__all__ = [
    "get_compressor_prompt",
    "COMPRESSOR_SYSTEM_PROMPT",
    "COMPRESSOR_USER_TEMPLATE",
    "REASONER_SYSTEM_PROMPT",
    "REASONER_USER_TEMPLATE",
]