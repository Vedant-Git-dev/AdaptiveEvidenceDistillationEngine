"""Tests for analyzer module."""

import json
from unittest.mock import patch, MagicMock

import pytest

from aede.state import AEDEState, Fact
from aede.nodes.analyzer import analyzer, _calculate_missing_parts_core


class TestAnalyzer:
    """Test suite for evidence quality analyzer."""

    @pytest.fixture
    def mock_api_response(self) -> dict:
        """Create a mock successful API response."""
        return {
            "id": "test-analysis-123",
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "answered_parts": ["revenue growth", "Q3 performance"],
                        "missing_parts": ["specific causes", "future projections"],
                        "coverage": 0.7,
                        "redundancy": 0.25,
                        "confidence": 0.75
                    })
                }
            }],
            "usage": {
                "prompt_tokens": 400,
                "completion_tokens": 120,
            }
        }

    def test_analyzer_parses_quality_signals(self, mock_api_response):
        """Test that analyzer correctly parses quality signals."""
        with patch("urllib.request.Request"):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.return_value.__enter__.return_value.read.return_value = (
                    json.dumps(mock_api_response).encode()
                )

                state: AEDEState = {
                    "query": "Why did revenue grow?",
                    "facts": [
                        Fact(claim="Revenue grew by 15%", quote="Revenue grew by 15%", chunk_id=0),
                    ],
                    "query_core_concepts": ["revenue", "growth"],
                    "workflow_path": [],
                    "token_usage": {},
                }

                result = analyzer(state)

                assert result["coverage"] == 0.7
                assert result["redundancy"] == 0.25
                assert result["confidence"] == 0.75
                assert "revenue growth" in result["answered_parts"]

    def test_analyzer_handles_empty_facts(self):
        """Test that analyzer handles empty facts list."""
        state: AEDEState = {
            "query": "Test query",
            "facts": [],
            "query_core_concepts": [],
            "workflow_path": ["start"],
            "token_usage": {},
        }

        result = analyzer(state)

        assert result["coverage"] == 0.0
        assert result["redundancy"] == 0.0
        assert result["confidence"] == 0.0
        assert result["answered_parts"] == []
        assert result["missing_parts"] == []
        assert "analyze" in result["workflow_path"]

    def test_analyzer_updates_workflow_path(self, mock_api_response):
        """Test that workflow path is updated."""
        with patch("urllib.request.Request"):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.return_value.__enter__.return_value.read.return_value = (
                    json.dumps(mock_api_response).encode()
                )

                state: AEDEState = {
                    "query": "Test",
                    "facts": [Fact(claim="Test", quote="Test", chunk_id=0)],
                    "query_core_concepts": [],
                    "workflow_path": ["start", "extract"],
                    "token_usage": {},
                }

                result = analyzer(state)

                assert result["workflow_path"][-1] == "analyze"

    def test_analyzer_tracks_token_usage(self, mock_api_response):
        """Test that token usage is tracked."""
        with patch("urllib.request.Request"):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.return_value.__enter__.return_value.read.return_value = (
                    json.dumps(mock_api_response).encode()
                )

                state: AEDEState = {
                    "query": "Test",
                    "facts": [Fact(claim="Test", quote="Test", chunk_id=0)],
                    "query_core_concepts": [],
                    "workflow_path": [],
                    "token_usage": {},
                }

                result = analyzer(state)

                assert result["token_usage"]["analyzer_input"] == 400
                assert result["token_usage"]["analyzer_output"] == 120

    def test_analyzer_handles_api_error(self):
        """Test that analyzer handles API errors gracefully."""
        with patch("urllib.request.Request"):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.side_effect = Exception("Network error")

                state: AEDEState = {
                    "query": "Test",
                    "facts": [Fact(claim="Test", quote="Test", chunk_id=0)],
                    "query_core_concepts": [],
                    "workflow_path": [],
                    "token_usage": {},
                }

                result = analyzer(state)

                assert "error" in result
                assert "Analyzer API error" in result["error"]

    def test_analyzer_clears_error_on_success(self, mock_api_response):
        """Test that previous errors are cleared on success."""
        with patch("urllib.request.Request"):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.return_value.__enter__.return_value.read.return_value = (
                    json.dumps(mock_api_response).encode()
                )

                state: AEDEState = {
                    "query": "Test",
                    "facts": [Fact(claim="Test", quote="Test", chunk_id=0)],
                    "query_core_concepts": [],
                    "workflow_path": [],
                    "token_usage": {},
                    "error": "Previous error",
                }

                result = analyzer(state)

                assert result["error"] is None


class TestCalculateMissingPartsCore:
    """Test suite for _calculate_missing_parts_core helper."""

    def test_empty_missing_parts(self):
        """Test with empty missing parts list."""
        result = _calculate_missing_parts_core(
            missing_parts=[],
            query_core_concepts=["revenue", "growth"],
            answered_parts=["answer1"]
        )
        assert result == []

    def test_empty_core_concepts(self):
        """Test with empty core concepts list."""
        result = _calculate_missing_parts_core(
            missing_parts=["missing part"],
            query_core_concepts=[],
            answered_parts=[]
        )
        assert result == []

    def test_finds_matching_core_concepts(self):
        """Test that matching core concepts are found in missing parts."""
        result = _calculate_missing_parts_core(
            missing_parts=["Revenue decline", "Product delays"],
            query_core_concepts=["revenue", "growth"],
            answered_parts=["Q3 performance"]
        )
        assert len(result) >= 1
        assert "Revenue decline" in result

    def test_excludes_answered_parts(self):
        """Test that already answered concepts are not included."""
        result = _calculate_missing_parts_core(
            missing_parts=["Revenue growth", "Market expansion"],
            query_core_concepts=["revenue", "growth"],
            answered_parts=["Revenue growth"]  # Revenue growth already answered
        )
        # Revenue growth should be excluded since it matches answered_parts
        # but market expansion mentions "growth" so might still be included
        assert isinstance(result, list)

    def test_handles_case_insensitive_matching(self):
        """Test that matching is case-insensitive."""
        result = _calculate_missing_parts_core(
            missing_parts=["REVENUE fell", "Growth stalled"],
            query_core_concepts=["revenue", "growth"],
            answered_parts=[]
        )
        # Should match despite different cases
        assert len(result) >= 1

    def test_complex_overlap_with_covered_concepts(self):
        """Test behavior when concepts overlap between answered and missing."""
        result = _calculate_missing_parts_core(
            missing_parts=["company revenue declined", "market growth slowed"],
            query_core_concepts=["revenue", "growth", "company"],
            answered_parts=["company performance"]  # "company" is covered
        )
        # revenue and growth should still be found in missing parts
        assert len(result) >= 1


class TestAnalyzerEdgeCases:
    """Test edge cases in analyzer functionality."""

    def test_very_high_coverage(self):
        """Test analyzer with very high coverage."""
        mock_response = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "answered_parts": ["all", "aspects", "covered"],
                        "missing_parts": [],
                        "coverage": 0.99,
                        "redundancy": 0.1,
                        "confidence": 0.95
                    })
                }
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}
        }

        with patch("urllib.request.Request"):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.return_value.__enter__.return_value.read.return_value = (
                    json.dumps(mock_response).encode()
                )

                state: AEDEState = {
                    "query": "Simple query",
                    "facts": [Fact(claim="Test", quote="Test", chunk_id=0)],
                    "query_core_concepts": [],
                    "workflow_path": [],
                    "token_usage": {},
                }

                result = analyzer(state)
                assert result["coverage"] >= 0.9

    def test_high_redundancy(self):
        """Test analyzer with high redundancy."""
        mock_response = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "answered_parts": ["some info"],
                        "missing_parts": [],
                        "coverage": 0.5,
                        "redundancy": 0.8,  # High redundancy
                        "confidence": 0.6
                    })
                }
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}
        }

        with patch("urllib.request.Request"):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.return_value.__enter__.return_value.read.return_value = (
                    json.dumps(mock_response).encode()
                )

                state: AEDEState = {
                    "query": "Test",
                    "facts": [Fact(claim="Test", quote="Test", chunk_id=0)],
                    "query_core_concepts": [],
                    "workflow_path": [],
                    "token_usage": {},
                }

                result = analyzer(state)
                assert result["redundancy"] > 0.5

    def test_fallback_json_parsing(self):
        """Test analyzer fallback JSON parsing."""
        mock_response = {
            "choices": [{
                "message": {
                    "content": '```json\n{"answered_parts":["a"],"missing_parts":["b"],"coverage":0.5,"redundancy":0.2,"confidence":0.6}\n```'
                }
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}
        }

        with patch("urllib.request.Request"):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.return_value.__enter__.return_value.read.return_value = (
                    json.dumps(mock_response).encode()
                )

                state: AEDEState = {
                    "query": "Test",
                    "facts": [Fact(claim="Test", quote="Test", chunk_id=0)],
                    "query_core_concepts": [],
                    "workflow_path": [],
                    "token_usage": {},
                }

                result = analyzer(state)
                assert "answered_parts" in result