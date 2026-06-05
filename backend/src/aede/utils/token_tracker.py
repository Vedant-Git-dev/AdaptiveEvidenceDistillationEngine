"""Token counting and tracking utilities."""

from typing import Dict, List, Optional
from dataclasses import dataclass, field

try:
    import tiktoken
except ImportError:
    tiktoken = None

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Settings


@dataclass
class TokenUsage:
    """Token usage record."""

    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int = field(init=False)

    def __post_init__(self):
        self.total_tokens = self.input_tokens + self.output_tokens


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken.

    Args:
        text: Text to count tokens for
        encoding_name: Tiktoken encoding name (default: cl100k_base for GPT-4/ChatGPT)

    Returns:
        Number of tokens
    """
    if tiktoken is None:
        raise ImportError("tiktoken is required. Install with: pip install tiktoken")

    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def count_tokens_for_model(text: str, model: str) -> int:
    """Count tokens for a specific model.

    Args:
        text: Text to count tokens for
        model: Model name (e.g., 'gpt-4', 'gemma-4-2b-it', 'gemini-2.5-flash')

    Returns:
        Number of tokens
    """
    # Map models to their encodings
    encoding_map = {
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "cl100k_base": "cl100k_base",
    }

    # Gemini uses cl100k_base compatible encoding
    if "gemini" in model.lower() or "gemma" in model.lower():
        encoding = encoding_map.get("gpt-4", "cl100k_base")
    else:
        encoding = encoding_map.get(model, "cl100k_base")

    return count_tokens(text, encoding)


def store_token_usage(
    state: dict,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """Store token usage in state for tracking.

    Args:
        state: State dictionary to update
        model: Model name used
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
    """
    if "token_usage" not in state:
        state["token_usage"] = {}

    usage_key = model
    counter = 1
    while usage_key in state["token_usage"]:
        counter += 1
        usage_key = f"{model}_{counter}"

    state["token_usage"][usage_key] = {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def get_total_tokens(state: dict, model: Optional[str] = None) -> int:
    """Get total tokens from state.

    Args:
        state: State dictionary containing token usage
        model: Optional model filter, if None sums all models

    Returns:
        Total token count
    """
    if "token_usage" not in state:
        return 0

    total = 0
    for key, usage in state["token_usage"].items():
        if model is None or usage.get("model") == model:
            total += usage.get("total_tokens", 0)

    return total


class TokenTracker:
    """Track token usage across operations."""

    def __init__(self):
        self._usage: List[TokenUsage] = []

    def add(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """Add token usage record."""
        self._usage.append(TokenUsage(model, input_tokens, output_tokens))

    def total_for_model(self, model: str) -> int:
        """Get total tokens for a specific model."""
        return sum(u.total_tokens for u in self._usage if u.model == model)

    def total_all(self) -> int:
        """Get total tokens across all models."""
        return sum(u.total_tokens for u in self._usage)

    def input_total(self) -> int:
        """Get total input tokens."""
        return sum(u.input_tokens for u in self._usage)

    def output_total(self) -> int:
        """Get total output tokens."""
        return sum(u.output_tokens for u in self._usage)

    def to_dict(self) -> Dict:
        """Export as dictionary."""
        return {
            "usage": [
                {"model": u.model, "input_tokens": u.input_tokens, "output_tokens": u.output_tokens}
                for u in self._usage
            ],
            "totals": {
                "all": self.total_all(),
                "input": self.input_total(),
                "output": self.output_total(),
            },
        }