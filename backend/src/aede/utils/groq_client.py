"""Groq API client wrapper for pipeline nodes."""

import json
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

        return {
            "text": response.choices[0].message.content if response.choices else "",
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
