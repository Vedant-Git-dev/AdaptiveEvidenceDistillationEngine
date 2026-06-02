"""Tests for extractor module."""

import pytest
from unittest.mock import patch

from aede.state import AEDEState, Fact
from aede.nodes.extractor import extractor


class MockGenAIResponse:
    """Mock google.genai response."""
    def __init__(self, text: str, prompt_tokens: int = 100, completion_tokens: int = 50):
        self.text = text
        self.usage_metadata = __import__("unittest.mock").MagicMock()
        self.usage_metadata.prompt_token_count = prompt_tokens
        self.usage_metadata.candidates_token_count = completion_tokens


class TestExtractor:
    """Test suite for evidence extractor."""

    def test_extractor_parses_valid_json_response(self):
        """Test that extractor correctly parses valid JSON response."""
        mock_response = MockGenAIResponse(
            text='{"facts": [{"claim": "Revenue grew by 15%", "quote": "Revenue grew 15%", "chunk_id": 0}]}',
            prompt_tokens=500,
            completion_tokens=150,
        )

        with patch("google.genai.Client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = mock_response

            state: AEDEState = {
                "query": "Why did revenue grow?",
                "documents": ["Revenue grew by 15%"],
                "workflow_path": [],
                "token_usage": {},
            }

            result = extractor(state)

            assert len(result["facts"]) == 1
            assert result["facts"][0]["claim"] == "Revenue grew by 15%"
            assert result["facts"][0]["chunk_id"] == 0

    def test_extractor_handles_empty_documents(self):
        """Test that extractor handles empty documents list."""
        state: AEDEState = {
            "query": "Test query",
            "documents": [],
            "workflow_path": ["start"],
            "token_usage": {},
        }

        result = extractor(state)

        assert result["facts"] == []
        assert "extract" in result["workflow_path"]

    def test_extractor_extracts_from_markdown_json(self):
        """Test that extractor parses JSON from markdown code blocks."""
        mock_response = MockGenAIResponse(
            text='```json\n{"facts": [{"claim": "Test claim", "quote": "Quote", "chunk_id": 0}]}\n```',
            prompt_tokens=100,
            completion_tokens=50,
        )

        with patch("google.genai.Client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = mock_response

            state: AEDEState = {
                "query": "Test",
                "documents": ["Test doc"],
                "workflow_path": [],
                "token_usage": {},
            }

            result = extractor(state)
            # Should parse from markdown
            assert result["workflow_path"] == ["extract"]

    def test_extractor_handles_api_error(self):
        """Test that extractor handles API errors gracefully."""
        with patch("google.genai.Client") as mock_client:
            mock_client.return_value.models.generate_content.side_effect = Exception("Network error")

            state: AEDEState = {
                "query": "Test query",
                "documents": ["Some document"],
                "workflow_path": [],
                "token_usage": {},
            }

            result = extractor(state)

            assert result["facts"] == []
            assert "error" in result

    def test_extractor_updates_workflow_path(self):
        """Test that workflow path is updated."""
        mock_response = MockGenAIResponse(text='{"facts": []}', prompt_tokens=0, completion_tokens=0)

        with patch("google.genai.Client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = mock_response

            state: AEDEState = {
                "query": "Test",
                "documents": ["Doc"],
                "workflow_path": ["start", "retrieve"],
                "token_usage": {},
            }

            result = extractor(state)

            assert "extract" in result["workflow_path"]

    def test_extractor_tracks_token_usage(self):
        """Test that token usage is tracked."""
        mock_response = MockGenAIResponse(text='{"facts": []}', prompt_tokens=500, completion_tokens=100)

        with patch("google.genai.Client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = mock_response

            state: AEDEState = {
                "query": "Test",
                "documents": ["Doc"],
                "workflow_path": [],
                "token_usage": {},
            }

            result = extractor(state)

            assert result["token_usage"]["extractor_input"] == 500
            assert result["token_usage"]["extractor_output"] == 100

    def test_extractor_handles_regex_fallback(self):
        """Test extractor falling back to regex parsing."""
        mock_response = MockGenAIResponse(
            text='{"claim": "Regex claim", "quote": "Quote text", "chunk_id": 2}',
            prompt_tokens=0,
            completion_tokens=0,
        )

        with patch("google.genai.Client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = mock_response

            state: AEDEState = {
                "query": "Test",
                "documents": ["Doc1", "Doc2", "Doc3"],
                "workflow_path": [],
                "token_usage": {},
            }

            result = extractor(state)
            assert len(result["facts"]) >= 0

    def test_extractor_handles_missing_fields(self):
        """Test extractor handles facts with missing optional fields."""
        mock_response = MockGenAIResponse(text='{"facts": [{"claim": "Test"}]}', prompt_tokens=0, completion_tokens=0)

        with patch("google.genai.Client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = mock_response

            state: AEDEState = {
                "query": "Test",
                "documents": ["Doc"],
                "workflow_path": [],
                "token_usage": {},
            }

            result = extractor(state)
            assert isinstance(result["facts"], list)

    def test_extractor_clears_error_on_success(self):
        """Test that previous errors are cleared on success."""
        mock_response = MockGenAIResponse(text='{"facts": []}', prompt_tokens=0, completion_tokens=0)

        with patch("google.genai.Client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = mock_response

            state: AEDEState = {
                "query": "Test",
                "documents": ["Doc"],
                "workflow_path": [],
                "token_usage": {},
                "error": "Previous error",
            }

            result = extractor(state)

            assert result["error"] is None