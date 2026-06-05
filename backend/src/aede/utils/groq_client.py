"""Groq API client wrapper for pipeline nodes."""

import json
import re
from typing import Optional

from aede.config import settings


def generate_with_groq(
    prompt: str,
    system_prompt: str = "",
    model: str = "",
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> dict:
    """Generate content using Groq API.

    Args:
        prompt: The user prompt
        system_prompt: System instructions
        model: Model to use (defaults to pipeline_model from config)
        temperature: Sampling temperature
        max_tokens: Max tokens to generate

    Returns:
        Dict with 'text' (response text) and 'usage' (token counts)
    """
    import os

    api_key = os.getenv("GROQ_API_KEY") or settings.models.groq_api_key
    if not api_key:
        return {"text": "", "usage": {}}

    model = model or settings.models.pipeline_model

    try:
        import httpx
        from groq import Groq

        # Use explicit httpx client (with timeout) to avoid proxies conflict
        client = Groq(
            api_key=api_key,
            http_client=httpx.Client(timeout=httpx.Timeout(60)),
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        raw_text = response.choices[0].message.content if response.choices else ""
        return {
            "text": _clean_model_output(raw_text),
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
        }

    except ImportError:
        print("WARNING: groq package not installed. Install with: pip install groq")
        return {"text": "", "usage": {}}
    except Exception as e:
        print(f"Groq API error: {e}")
        return {"text": "", "usage": {}, "error": str(e)}


def _clean_model_output(text: str) -> str:
    """Strip reasoning/fence wrappers that small models sometimes emit.

    Qwen3-32B (and similar reasoning models) wrap their final answer in two
    ways the JSON parsers in this pipeline can't handle:
      1. A leading <think>...</think> block (the model's chain of thought).
      2. A ```json ... ``` code fence around the answer, even when the
         system prompt asked for raw JSON.

    The prompt for each node already says "Return ONLY valid JSON" — but the
    model ignores that on roughly half of calls. Stripping both shapes here
    keeps the call sites (extractor / analyzer / compressor / small_reasoner)
    from needing to know which model is in use.
    """
    if not text:
        return text

    # 1. Drop the <think>...</think> block. Use DOTALL so the think body can
    #    span newlines. Keep whatever comes after.
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # 2. If the response is wrapped in a ```json ... ``` or ``` ... ``` fence,
    #    keep only the inner content.
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)

    return cleaned.strip()
