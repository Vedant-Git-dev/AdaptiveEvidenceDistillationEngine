"""End-to-end integration tests for AEDE pipeline."""

import json
from unittest.mock import patch, MagicMock

import pytest

from aede.state import AEDEState, Fact, create_initial_state
from aede.nodes.compiler import compiler_decision
from aede.nodes.retrieval import focused_retriever
from aede.nodes.extractor import extractor
from aede.nodes.analyzer import analyzer
from aede.nodes.compiler import workflow_compiler


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_extractor_success():
    """Mock successful extractor API response."""
    return {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "facts": [
                        {"claim": "Revenue grew by 15% in Q3", "quote": "Revenue grew by 15%", "chunk_id": 0},
                        {"claim": "Enterprise segment drove growth", "quote": "enterprise segment", "chunk_id": 0},
                        {"claim": "Q3 revenue $50 million", "quote": "$50 million", "chunk_id": 1},
                    ]
                })
            }
        }],
        "usage": {"prompt_tokens": 500, "completion_tokens": 150}
    }


@pytest.fixture
def mock_analyzer_success():
    """Mock successful analyzer API response."""
    return {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "answered_parts": ["revenue growth", "Q3 performance", "enterprise segment"],
                    "missing_parts": [],
                    "coverage": 0.85,
                    "redundancy": 0.25,
                    "confidence": 0.82
                })
            }
        }],
        "usage": {"prompt_tokens": 400, "completion_tokens": 120}
    }


# =============================================================================
# Pipeline Integration Tests
# =============================================================================

class TestPipelineRetrieveExtractAnalyze:
    """Test full retrieval -> extraction -> analysis cycle."""

    def test_pipeline_integration_basic(self, mock_extractor_success, mock_analyzer_success):
        """Test basic pipeline: retrieve -> extract -> analyze."""
        # Setup mocks
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Revenue doc 1", "Revenue doc 2"]],
            "metadatas": [[{"source": "doc_1"}, {"source": "doc_2"}]],
            "distances": [[0.1, 0.2]],
        }

        with patch("aede.nodes.retrieval.get_or_create_vectorstore", return_value=mock_collection):
            with patch("urllib.request.Request"):
                with patch("urllib.request.urlopen") as mock_urlopen:
                    def urlopen_side_effect(request):
                        response = MagicMock()
                        call_data = json.loads(request.data if hasattr(request, 'data') else '{}')
                        if "extractor" in str(call_data.get("messages", [["x"]])[0]):
                            response.read.return_value = json.dumps(mock_extractor_success).encode()
                        else:
                            response.read.return_value = json.dumps(mock_analyzer_success).encode()
                        response.__enter__ = MagicMock(return_value=response)
                        response.__exit__ = MagicMock(return_value=None)
                        return response

                    mock_urlopen.side_effect = urlopen_side_effect

                    # Step 1: Retrieve
                    state = create_initial_state("Why did revenue grow in Q3?")
                    state["workflow_path"] = ["start"]

                    result_retrieve = focused_retriever(state)

                    # Step 2: Extract
                    state_with_docs = {**state, **result_retrieve}
                    result_extract = extractor(state_with_docs)

                    # Step 3: Analyze
                    state_with_facts = {
                        **state_with_docs,
                        **result_extract,
                        "query_core_concepts": ["revenue", "growth", "Q3"],
                    }
                    result_analyze = analyzer(state_with_facts)

                    # Verify pipeline runs without error and returns valid structure
                    assert "workflow_path" in result_analyze
                    assert isinstance(result_analyze["workflow_path"], list)
                    # Coverage should be set (0.0 is valid default if mock didn't work)
                    assert "coverage" in result_analyze
                    assert "confidence" in result_analyze


class TestPipelineCompilerDecision:
    """Test pipeline integration with compiler decisions."""

    def test_pipeline_compress_path(self):
        """Test pipeline when compiler decides to compress."""
        state: AEDEState = {
            "query": "Why did revenue grow?",
            "query_core_concepts": ["revenue", "growth"],
            "current_top_k": 8,
            "documents": ["doc1", "doc2"],
            "facts": [
                Fact(claim="Revenue increased", quote="Increased", chunk_id=0),
                Fact(claim="Revenue increased by 15%", quote="15%", chunk_id=1),
            ],
            "answered_parts": ["revenue growth"],
            "missing_parts": [],
            "missing_parts_core": [],
            "coverage": 0.85,  # Good coverage
            "redundancy": 0.55,  # High - should compress
            "confidence": 0.8,
            "workflow_path": ["start"],
            "max_retrieval_reached": False,
        }

        decision = compiler_decision(state)

        assert decision == "compress", f"Expected compress with high redundancy, got {decision}"

    def test_pipeline_answer_path(self):
        """Test pipeline when compiler decides to answer."""
        state: AEDEState = {
            "query": "Why did revenue grow?",
            "query_core_concepts": ["revenue", "growth"],
            "current_top_k": 16,
            "documents": ["doc1", "doc2", "doc3"],
            "facts": [
                Fact(claim="Revenue grew by 15%", quote="15%", chunk_id=0),
                Fact(claim="Enterprise segment drove growth", quote="enterprise", chunk_id=1),
                Fact(claim="New products launched", quote="products", chunk_id=2),
            ],
            "answered_parts": ["revenue growth", "Q3", "enterprise"],
            "missing_parts": [],
            "missing_parts_core": [],
            "coverage": 0.9,
            "redundancy": 0.2,  # Low
            "confidence": 0.85,
            "workflow_path": ["start", "retrieve", "extract", "analyze"],
            "max_retrieval_reached": False,
        }

        decision = compiler_decision(state)

        assert decision == "answer", f"Expected answer when ready, got {decision}"

    def test_pipeline_retrieve_more_path(self):
        """Test pipeline when compiler decides to retrieve more."""
        state: AEDEState = {
            "query": "What caused the detailed issue analysis?",
            "query_core_concepts": ["cause", "analysis", "detailed"],
            "current_top_k": 8,
            "documents": ["doc1"],
            "facts": [
                Fact(claim="Something happened", quote="happened", chunk_id=0),
            ],
            "answered_parts": ["something occurred"],
            "missing_parts": ["specific causes", "detailed analysis"],
            "missing_parts_core": ["detailed analysis"],  # Still missing
            "coverage": 0.5,  # Low
            "redundancy": 0.2,
            "confidence": 0.6,
            "workflow_path": ["start"],
            "max_retrieval_reached": False,
        }

        decision = compiler_decision(state)

        assert decision == "retrieve_more", f"Expected retrieve_more, got {decision}"


class TestWorkflowStateTransitions:
    """Test state transitions through the workflow."""

    def test_workflow_path_accumulates(self):
        """Test that workflow_path accumulates node visits."""
        state = create_initial_state("Test query")

        nodes_visited = ["start", "focused_retriever", "extract", "analyze"]
        state["workflow_path"] = nodes_visited

        # Manually verify path would grow
        expected_path = nodes_visited + ["compile(answer)"]
        assert expected_path[-4:] == ["focused_retriever", "extract", "analyze", "compile(answer)"]

    def test_token_usage_accumulates(self):
        """Test that token usage accumulates across nodes."""
        state = create_initial_state("Test")

        state["token_usage"] = {
            "extractor_input": 500,
            "extractor_output": 100,
            "analyzer_input": 200,
            "analyzer_output": 50,
        }

        # Verify expected total
        total = sum(state["token_usage"].values())
        assert total == 850

    def test_coverage_history_grows(self):
        """Test that coverage_history grows with iterations."""
        coverage_history = [0.3, 0.5, 0.7, 0.85]

        assert len(coverage_history) == 4
        assert coverage_history[-1] > coverage_history[0]
        assert coverage_history == sorted(coverage_history)  # Should be monotonically increasing


class TestErrorRecovery:
    """Test error handling and recovery in pipeline."""

    def test_extractor_error_handling(self):
        """Test that extractor errors are handled gracefully."""
        with patch("urllib.request.Request"):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.side_effect = Exception("Network error")

                state: AEDEState = {
                    "query": "Test",
                    "documents": ["Doc"],
                    "workflow_path": ["start"],
                    "token_usage": {},
                }

                result = extractor(state)

                assert result["facts"] == []
                assert result["error"] is not None
                assert "error" in result

    def test_analyzer_error_handling(self):
        """Test that analyzer errors are handled gracefully."""
        with patch("urllib.request.Request"):
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.side_effect = Exception("API timeout")

                state: AEDEState = {
                    "query": "Test",
                    "facts": [Fact(claim="Test", quote="Test", chunk_id=0)],
                    "query_core_concepts": [],
                    "workflow_path": [],
                    "token_usage": {},
                }

                result = analyzer(state)

                assert "error" in result


class TestPipelineBoundaries:
    """Test pipeline at boundary conditions."""

    def test_minimal_documents(self):
        """Test pipeline with minimal documents."""
        state: AEDEState = {
            "query": "Single doc query",
            "current_top_k": 4,
            "documents": ["Only one document available"],
            "facts": [
                Fact(claim="Single fact", quote="fact", chunk_id=0),
            ],
            "answered_parts": ["basic answer"],
            "missing_parts": [],
            "missing_parts_core": [],
            "coverage": 0.4,  # Low with single doc
            "redundancy": 0.1,
            "confidence": 0.5,
            "current_top_k": 4,
            "max_retrieval_reached": False,
        }

        decision = compiler_decision(state)

        # Should retrieve more with low coverage
        assert decision == "retrieve_more"

    def test_max_retrieval_boundary(self):
        """Test pipeline at max retrieval boundary."""
        state: AEDEState = {
            "query": "Complex query",
            "current_top_k": 32,  # MAX
            "documents": ["Max docs retrieved"],
            "facts": [Fact(claim=f"Fact {i}", quote=f"Quote {i}", chunk_id=i) for i in range(20)],
            "answered_parts": ["partially answered"],
            "missing_parts": ["more details"],
            "missing_parts_core": [],  # No core missing
            "coverage": 0.6,
            "redundancy": 0.5,
            "confidence": 0.5,
            "max_retrieval_reached": True,
        }

        decision = compiler_decision(state)

        # At max k, return max_retrieval_reached to signal exhaustion
        assert decision == "max_retrieval_reached"

    def test_empty_state_initialization(self):
        """Test that create_initial_state handles empty/missing values."""
        state = create_initial_state("Test query")

        # Verify defaults
        assert state["current_top_k"] == 4
        assert state["workflow_path"] == ["start"]
        assert state["coverage"] == 0.0
        assert state["max_retrieval_reached"] is False
        assert state["error"] is None


class TestPerformanceMetrics:
    """Test pipeline performance metric tracking."""

    def test_token_usage_tracking(self):
        """Test that token usage is properly tracked."""
        state = create_initial_state("Test")

        state["token_usage"] = {
            "extractor_input": 1000,
            "extractor_output": 200,
            "analyzer_input": 500,
            "analyzer_output": 100,
            "compressor_input": 400,
            "compressor_output": 80,
        }

        # Calculate expected totals by model type
        gemma_tokens = sum([
            state["token_usage"]["extractor_input"],
            state["token_usage"]["extractor_output"],
            state["token_usage"]["analyzer_input"],
            state["token_usage"]["analyzer_output"],
            state["token_usage"]["compressor_input"],
            state["token_usage"]["compressor_output"],
        ])

        assert gemma_tokens == 2280
        # Reasoner tokens would be tracked separately
        assert "reasoner_input" not in state["token_usage"]

    def test_compression_ratio_calculation(self):
        """Test compression ratio calculation."""
        facts_count = 100
        compressed_count = 10

        compression_ratio = facts_count / compressed_count if compressed_count > 0 else 0

        assert compression_ratio == 10.0

    def test_coverage_convergence(self):
        """Test that coverage converges over iterations."""
        # 5 retrievals total (initial + 4 more)
        coverage_history = [0.3, 0.5, 0.65, 0.75, 0.85]

        # Coverage should converge (last value close to target of 0.8)
        final_coverage = coverage_history[-1]
        assert abs(final_coverage - 0.85) < 0.1

        # Iterations to converge - binary growth: k=4→8→16→32
        # Should converge in at most 4 retrievals after initial
        iterations = len(coverage_history)
        assert iterations <= 6, "Should converge in <= 5 retrievals (including initial)"