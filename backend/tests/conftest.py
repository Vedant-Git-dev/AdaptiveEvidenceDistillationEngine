"""Pytest fixtures and utilities for AEDE testing."""

from typing import Generator
from unittest.mock import MagicMock, patch
import json

import pytest

from aede.state import AEDEState, Fact, create_initial_state


# =============================================================================
# Sample Documents
# =============================================================================

SAMPLE_DOCUMENTS = [
    "Revenue grew by 15% in Q3 due to increased sales in the enterprise segment.",
    "Q3 revenue reached $50 million, up from $43 million in Q2.",
    "CEO stated that the company's focus on enterprise customers has been successful.",
    "The company launched three new products in Q3 targeting the enterprise market.",
]

SAMPLE_FACTS: list[Fact] = [
    Fact(claim="Revenue grew by 15% in Q3", quote="Revenue grew by 15% in Q3", chunk_id=0),
    Fact(claim="Growth driven by enterprise segment", quote="increased sales in the enterprise segment", chunk_id=0),
    Fact(claim="Q3 revenue was $50 million", quote="Q3 revenue reached $50 million", chunk_id=1),
    Fact(claim="Revenue up from $43 million in Q2", quote="up from $43 million in Q2", chunk_id=1),
]

SAMPLE_QUERY = "Why did revenue grow in Q3?"


# =============================================================================
# State Factories
# =============================================================================

@pytest.fixture
def initial_state() -> AEDEState:
    """Create initial state for testing."""
    return create_initial_state(query=SAMPLE_QUERY)


@pytest.fixture
def state_with_documents(initial_state: AEDEState) -> AEDEState:
    """State with retrieved documents."""
    return {**initial_state, "documents": SAMPLE_DOCUMENTS, "workflow_path": ["start", "retrieve"]}


@pytest.fixture
def state_with_facts(state_with_documents: AEDEState) -> AEDEState:
    """State with extracted facts."""
    return {**state_with_documents, "facts": SAMPLE_FACTS, "workflow_path": state_with_documents["workflow_path"] + ["extract"]}


@pytest.fixture
def state_with_quality_signals(state_with_facts: AEDEState) -> AEDEState:
    """State with quality analysis signals."""
    return {
        **state_with_facts,
        "answered_parts": ["revenue growth", "Q3 performance"],
        "missing_parts": ["specific causes"],
        "missing_parts_core": [],
        "coverage": 0.65,
        "redundancy": 0.3,
        "confidence": 0.7,
        "query_core_concepts": ["revenue", "growth", "Q3"],
        "workflow_path": state_with_facts["workflow_path"] + ["analyze"],
    }


@pytest.fixture
def state_ready_to_answer() -> AEDEState:
    """State ready for final answer."""
    return {
        "query": SAMPLE_QUERY,
        "query_core_concepts": ["revenue", "growth", "Q3"],
        "current_top_k": 8,
        "documents": SAMPLE_DOCUMENTS[:2],
        "facts": SAMPLE_FACTS[:3],
        "compressed_evidence": [
            "Revenue grew by 15% in Q3",
            "Enterprise segment drove the growth",
        ],
        "answered_parts": ["revenue growth", "enterprise segment"],
        "missing_parts": [],
        "missing_parts_core": [],
        "coverage": 0.85,
        "redundancy": 0.2,
        "confidence": 0.85,
        "workflow_path": ["start", "retrieve", "extract", "analyze", "compress"],
        "token_usage": {},
        "answer": "",
        "max_retrieval_reached": False,
        "error": None,
    }


@pytest.fixture
def state_max_retrieval() -> AEDEState:
    """State at max retrieval."""
    return {
        "query": "Complex query",
        "query_core_concepts": ["complex", "analysis"],
        "current_top_k": 32,
        "documents": SAMPLE_DOCUMENTS * 4,
        "facts": SAMPLE_FACTS * 2,
        "answered_parts": ["basic info"],
        "missing_parts": [],
        "missing_parts_core": [],
        "coverage": 0.5,
        "redundancy": 0.6,
        "confidence": 0.4,
        "workflow_path": ["start"],
        "token_usage": {},
        "answer": "",
        "max_retrieval_reached": True,
        "error": None,
    }


# =============================================================================
# Mock Classes
# =============================================================================

class MockCollection:
    """Mock ChromaDB collection."""

    def __init__(self, documents: list[str] | None = None):
        self._documents = documents or SAMPLE_DOCUMENTS
        self.query_calls: list[dict] = []

    def query(self, query_texts: list[str], n_results: int, **kwargs) -> dict:
        self.query_calls.append({"query_texts": query_texts, "n_results": n_results, "kwargs": kwargs})
        docs = self._documents[:n_results]
        return {
            "documents": [docs],
            "metadatas": [[{"source": f"doc_{i}"} for i in range(len(docs))]],
            "distances": [[0.1 * i for i in range(len(docs))]],
        }

    def add(self, **kwargs):
        pass


class MockGenAIResponse:
    """Mock google.genai Client response."""

    def __init__(self, text: str, prompt_tokens: int = 100, completion_tokens: int = 50):
        self.text = text
        self.usage_metadata = MagicMock()
        self.usage_metadata.prompt_token_count = prompt_tokens
        self.usage_metadata.candidates_token_count = completion_tokens


# =============================================================================
# Mock API Responses for google.genai
# =============================================================================

@pytest.fixture
def mock_extractor_response() -> MagicMock:
    """Mock Gemini API response for extractor."""
    return MockGenAIResponse(
        text='{"facts": [{"claim": "Revenue grew by 15%", "quote": "Revenue grew by 15%", "chunk_id": 0}]}',
        prompt_tokens=500,
        completion_tokens=100,
    )


@pytest.fixture
def mock_analyzer_response() -> MagicMock:
    """Mock Gemini API response for analyzer."""
    return MockGenAIResponse(
        text='{"answered_parts": ["revenue growth"], "missing_parts": [], "coverage": 0.85, "redundancy": 0.2, "confidence": 0.82}',
        prompt_tokens=300,
        completion_tokens=80,
    )


# =============================================================================
# Patches for google.genai
# =============================================================================

def patch_retrieval() -> Generator[MagicMock, None, None]:
    """Patch retrieval to use mock collection."""
    with patch("aede.nodes.retrieval.get_or_create_vectorstore") as mock_get:
        mock_get.return_value = MockCollection()
        yield mock_get


def patch_extractor_api(mock_extractor_response: MagicMock) -> Generator[MagicMock, None, None]:
    """Patch google.genai.Client for extractor."""
    with patch("google.genai.Client") as mock_client:
        mock_client.return_value.models.generate_content.return_value = mock_extractor_response
        yield mock_client


def patch_analyzer_api(mock_analyzer_response: MagicMock) -> Generator[MagicMock, None, None]:
    """Patch google.genai.Client for analyzer."""
    with patch("google.genai.Client") as mock_client:
        mock_client.return_value.models.generate_content.return_value = mock_analyzer_response
        yield mock_client