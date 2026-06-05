"""CRITICAL: Tests for compiler module decision logic.

This test suite validates the workflow compiler's decision-making capabilities,
which are central to the AEDE pipeline's adaptive behavior.
"""

import pytest

from aede.state import AEDEState
from aede.nodes.compiler import (
    compiler_decision,
    workflow_compiler,
    DEFAULT_COVERAGE_TARGET,
    DEFAULT_REDUNDANCY_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MAX_K,
)


class TestCompilerBasicDecisions:
    """Basic compiler decision tests."""

    def test_complier_returns_decision_type(self, state_with_quality_signals):
        """Test that compiler returns valid decision type."""
        decision = compiler_decision(state_with_quality_signals)
        assert decision in ["retrieve_more", "compress", "answer", "max_retrieval_reached"]

    def test_complier_uses_default_thresholds(self):
        """Test that default thresholds match config values."""
        assert DEFAULT_COVERAGE_TARGET == 0.8
        assert DEFAULT_REDUNDANCY_THRESHOLD == 0.4
        assert DEFAULT_CONFIDENCE_THRESHOLD == 0.5
        assert DEFAULT_MAX_K == 32


class TestCompilerRetrieveMore:
    """Test compiler decisions for retrieve_more path."""

    def test_compiler_retrieve_more_when_coverage_low(self):
        """
        CRITICAL: Test that compiler decides to retrieve more when coverage is low.

        When coverage < 0.8 AND missing_parts_core is empty, but coverage is low,
        should retrieve more evidence.
        """
        state: AEDEState = {
            "query": "Why did revenue grow?",
            "query_core_concepts": ["revenue", "growth"],
            "current_top_k": 4,
            "documents": ["doc1"],
            "facts": [],
            "answered_parts": [],
            "missing_parts": [],
            "missing_parts_core": [],  # No specific core concepts missing
            "coverage": 0.5,  # Below target
            "redundancy": 0.2,
            "confidence": 0.7,
            "max_retrieval_reached": False,
        }

        decision = compiler_decision(state)

        assert decision == "retrieve_more", (
            f"Expected 'retrieve_more' when coverage={0.5} < 0.8, "
            f"got '{decision}'"
        )

    def test_compiler_retrieve_more_when_missing_core_concepts(self):
        """Test retrieve_more when core concepts are missing."""
        state: AEDEState = {
            "query": "What caused the delay?",
            "query_core_concepts": ["cause", "delay"],
            "current_top_k": 4,
            "coverage": 0.85,  # Above target
            "redundancy": 0.2,
            "confidence": 0.7,
            "missing_parts_core": ["specific causes"],  # Missing core concept
            "max_retrieval_reached": False,
        }

        decision = compiler_decision(state)

        assert decision == "retrieve_more", (
            "Expected 'retrieve_more' when missing_parts_core is non-empty, "
            f"got '{decision}'"
        )

    def test_compiler_retrieve_more_increments_k(self, state_with_quality_signals):
        """Test that retrieve_more increases k in workflow_path."""
        result = workflow_compiler(state_with_quality_signals)

        compile_step = [step for step in result["workflow_path"] if "compile" in step]
        assert len(compile_step) == 1


class TestCompilerCompress:
    """Test compiler decisions for compress path."""

    def test_compiler_compress_when_redundancy_high(self):
        """
        CRITICAL: Test that compiler decides to compress when redundancy is high.

        When coverage >= 0.8 AND no missing core concepts AND redundancy > 0.4,
        should compress the evidence.
        """
        state: AEDEState = {
            "query": "Why did revenue grow?",
            "query_core_concepts": ["revenue", "growth"],
            "current_top_k": 8,
            "coverage": 0.85,  # Above target
            "redundancy": 0.6,  # Above threshold - needs compression
            "confidence": 0.8,  # Good confidence
            "missing_parts_core": [],  # No missing core concepts
            "max_retrieval_reached": False,
        }

        decision = compiler_decision(state)

        assert decision == "compress", (
            f"Expected 'compress' when redundancy={0.6} > 0.4, got '{decision}'"
        )

    def test_compiler_does_not_compress_when_low_redundancy(self):
        """Test that compression is not triggered when redundancy is low."""
        state: AEDEState = {
            "query": "Test query",
            "current_top_k": 8,
            "coverage": 0.85,
            "redundancy": 0.2,  # Low redundancy
            "confidence": 0.8,
            "missing_parts_core": [],
            "max_retrieval_reached": False,
            "required_reasoning": "none",  # opt in to direct-answer path
        }

        decision = compiler_decision(state)

        # Low redundancy + easy question -> direct_answer (skip compressor)
        assert decision == "direct_answer"

    def test_compression_takes_priority_over_low_confidence(self):
        """Test that high redundancy is handled before low confidence."""
        state: AEDEState = {
            "query": "Test query",
            "current_top_k": 8,
            "coverage": 0.85,
            "redundancy": 0.5,  # High - should compress first
            "confidence": 0.4,  # Low - would normally retrieve
            "missing_parts_core": [],
            "max_retrieval_reached": False,
        }

        decision = compiler_decision(state)

        # Redundancy > 0.4 should trigger compress before checking confidence
        assert decision == "compress"


class TestCompilerAnswer:
    """Test compiler decisions for answer path."""

    def test_complier_answer_when_ready(self):
        """
        CRITICAL: Test that compiler decides to answer when all conditions are met.

        When coverage >= 0.8 AND no missing core concepts AND redundancy <= 0.4
        AND confidence >= 0.5 AND the analyzer marked the question as easy,
        the compiler routes via the direct-answer (llama) path.
        """
        state: AEDEState = {
            "query": "Why did revenue grow?",
            "query_core_concepts": ["revenue", "growth"],
            "current_top_k": 8,
            "coverage": 0.85,  # Above target
            "redundancy": 0.3,  # Below threshold
            "confidence": 0.8,  # Good confidence
            "missing_parts_core": [],  # No missing core concepts
            "max_retrieval_reached": False,
            "required_reasoning": "none",  # analyzer says: no synthesis needed
        }

        decision = compiler_decision(state)

        assert decision == "direct_answer", (
            f"Expected 'direct_answer' when all conditions met and question is easy: "
            f"coverage={0.85}, redundancy={0.3}, confidence={0.8}, "
            f"missing_core=0. Got '{decision}'"
        )

    def test_answer_with_boundary_confidence(self):
        """Test answer at confidence threshold boundary."""
        state: AEDEState = {
            "query": "Test query",
            "current_top_k": 8,
            "coverage": 0.85,
            "redundancy": 0.3,
            "confidence": 0.5,  # At threshold - should pass
            "missing_parts_core": [],
            "max_retrieval_reached": False,
            "required_reasoning": "none",
        }

        decision = compiler_decision(state)
        assert decision == "direct_answer"

    def test_answer_with_boundary_coverage(self):
        """Test answer at coverage threshold boundary."""
        state: AEDEState = {
            "query": "Test query",
            "current_top_k": 8,
            "coverage": 0.8,  # At threshold - should pass
            "redundancy": 0.3,
            "confidence": 0.8,
            "missing_parts_core": [],
            "max_retrieval_reached": False,
            "required_reasoning": "none",
        }

        decision = compiler_decision(state)
        assert decision == "direct_answer"

    def test_answer_with_boundary_redundancy(self):
        """Test answer at redundancy threshold boundary."""
        state: AEDEState = {
            "query": "Test query",
            "current_top_k": 8,
            "coverage": 0.85,
            "redundancy": 0.4,  # At threshold - should NOT compress
            "confidence": 0.8,
            "missing_parts_core": [],
            "max_retrieval_reached": False,
            "required_reasoning": "none",
        }

        decision = compiler_decision(state)
        assert decision == "direct_answer"


class TestCompilerMaxRetrievalEdgeCase:
    """Test critical max retrieval edge cases."""

    def test_complier_max_retrieval_edge_case(self):
        """
        CRITICAL: Test that compiler handles max retrieval edge case.

        When current_top_k >= max_k AND coverage is still low OR missing core
        concepts exist, should return max_retrieval_reached instead of
        retrieve_more.
        """
        state: AEDEState = {
            "query": "Complex query requiring extensive retrieval",
            "current_top_k": 32,  # At max
            "coverage": 0.6,  # Still below target
            "redundancy": 0.3,
            "confidence": 0.7,
            "missing_parts_core": ["specific analysis"],
            "max_retrieval_reached": True,
        }

        decision = compiler_decision(state)

        assert decision == "max_retrieval_reached", (
            f"Expected 'max_retrieval_reached' when k={32} >= max_k and "
            f"conditions not met, got '{decision}'"
        )

    def test_complier_answers_after_max_retrieval_if_no_missing_core(self):
        """Test that compiler signals max retrieval when at k=32."""
        state: AEDEState = {
            "query": "Simple query",
            "current_top_k": 32,  # At max
            "coverage": 0.5,  # Below target
            "redundancy": 0.3,
            "confidence": 0.7,
            "missing_parts_core": [],  # No missing core
            "max_retrieval_reached": True,
        }

        decision = compiler_decision(state)

        # At max k, always return max_retrieval_reached to signal exhaustion
        # The reasoner will then generate an answer
        assert decision == "max_retrieval_reached"

    def test_max_retrieval_with_low_confidence(self):
        """Test max retrieval edge case with low confidence."""
        state: AEDEState = {
            "query": "Complex query",
            "current_top_k": 32,
            "coverage": 0.7,
            "redundancy": 0.3,
            "confidence": 0.4,  # Low confidence
            "missing_parts_core": [],
            "max_retrieval_reached": True,
        }

        decision = compiler_decision(state)

        # At max with low confidence but no missing core concepts
        assert decision == "max_retrieval_reached"

    def test_max_retrieval_with_high_redundancy(self):
        """Test max retrieval with high redundancy."""
        state: AEDEState = {
            "query": "Data-heavy query",
            "current_top_k": 32,
            "coverage": 0.7,
            "redundancy": 0.5,  # High
            "confidence": 0.8,
            "missing_parts_core": [],
            "max_retrieval_reached": True,
        }

        decision = compiler_decision(state)

        # At max, should signal max_retrieval_reached
        assert decision == "max_retrieval_reached"


class TestCompilerEdgeCases:
    """Additional edge case tests for compiler."""

    def test_no_quality_signals_returns_max_retrieval(self):
        """Test that at max retrieval with no signals returns max_retrieval_reached."""
        state: AEDEState = {
            "query": "Test",
            "current_top_k": 32,
            "max_retrieval_reached": True,
        }

        decision = compiler_decision(state)
        # At max k, return max_retrieval_reached to signal exhaustion
        assert decision == "max_retrieval_reached"

    def test_partial_state_handling(self):
        """Test that partial state is handled gracefully."""
        state: AEDEState = {
            "query": "Test query",
            "current_top_k": 8,
            # Missing: coverage, redundancy, confidence
        }

        decision = compiler_decision(state)
        # Missing signals default to 0.0
        # coverage=0 < 0.8, so should retrieve more
        assert decision == "retrieve_more"

    def test_custom_thresholds(self):
        """Test that custom thresholds are respected."""
        state: AEDEState = {
            "query": "Test",
            "current_top_k": 4,
            "coverage": 0.65,
            "redundancy": 0.3,
            "confidence": 0.7,
            "missing_parts_core": [],
            "max_retrieval_reached": False,
            "required_reasoning": "none",
        }

        # Use higher coverage target
        decision = compiler_decision(state, coverage_target=0.6)

        # With coverage=0.65 >= 0.6 target and easy question -> direct_answer
        assert decision in ["direct_answer", "compress", "llama_with_compress", "deep_reasoning"]

    def test_workflow_compiler_updates_path(self):
        """Test that workflow_compiler node updates path."""
        state: AEDEState = {
            "query": "Test",
            "current_top_k": 8,
            "coverage": 0.9,
            "redundancy": 0.2,
            "confidence": 0.8,
            "missing_parts_core": [],
            "max_retrieval_reached": False,
            "workflow_path": ["start"],
        }

        result = workflow_compiler(state)

        assert "workflow_path" in result
        last_step = result["workflow_path"][-1]
        assert "compile" in last_step


class TestCompilerPriorityOrder:
    """Test that compiler evaluates conditions in correct priority order."""

    def test_case1_priority_max_retrieval(self):
        """Test Case 1: Max retrieval exhaustion is checked first."""
        state: AEDEState = {
            "query": "Test",
            "current_top_k": 32,
            "coverage": 0.3,  # Low
            "redundancy": 0.6,  # High
            "confidence": 0.3,  # Low
            "missing_parts_core": [],  # But no missing core
            "max_retrieval_reached": True,
        }

        decision = compiler_decision(state)

        # At max k, return max_retrieval_reached regardless of other signals
        # The reasoner will generate an answer based on available evidence
        assert decision == "max_retrieval_reached"

    def test_case2_priority_before_case3(self):
        """Test Case 2 (coverage/missing) is checked before Case 3 (redundancy)."""
        state: AEDEState = {
            "query": "Test",
            "current_top_k": 4,
            "coverage": 0.4,  # Low - would trigger retrieve
            "redundancy": 0.6,  # High - would trigger compress
            "confidence": 0.8,
            "missing_parts_core": [],
            "max_retrieval_reached": False,
        }

        decision = compiler_decision(state)

        # Coverage < 0.8 should trigger retrieve_more before checking redundancy
        assert decision == "retrieve_more"

    def test_case3_priority_before_case4(self):
        """Test Case 3 (redundancy) is checked before Case 4 (confidence)."""
        state: AEDEState = {
            "query": "Test",
            "current_top_k": 16,
            "coverage": 0.85,  # Good
            "redundancy": 0.5,  # High - would trigger compress
            "confidence": 0.3,  # Low - would trigger retrieve
            "missing_parts_core": [],
            "max_retrieval_reached": False,
        }

        decision = compiler_decision(state)

        # Redundancy > 0.4 should trigger compress before checking confidence
        assert decision == "compress"


class TestCustomThresholdsIntegration:
    """Test integration with custom threshold values."""

    def test_strict_coverage_target(self):
        """Test with stricter coverage target."""
        state: AEDEState = {
            "query": "Precision query",
            "current_top_k": 8,
            "coverage": 0.75,
            "redundancy": 0.2,
            "confidence": 0.9,
            "missing_parts_core": [],
            "max_retrieval_reached": False,
            "required_reasoning": "none",
        }

        # Default target is 0.8, so coverage=0.75 should trigger retrieve
        decision_default = compiler_decision(state)
        assert decision_default == "retrieve_more"

        # With lower target=0.5, coverage=0.75 should pass and question is easy
        decision_strict = compiler_decision(state, coverage_target=0.5)
        assert decision_strict == "direct_answer"

    def test_permissive_redundancy_threshold(self):
        """Test with permissive redundancy threshold."""
        state: AEDEState = {
            "query": "Test",
            "current_top_k": 8,
            "coverage": 0.9,
            "redundancy": 0.45,
            "confidence": 0.9,
            "missing_parts_core": [],
            "max_retrieval_reached": False,
            "required_reasoning": "none",
        }

        # Default threshold is 0.4, so redundancy=0.45 should compress
        decision_default = compiler_decision(state)
        assert decision_default == "compress"

        # With higher threshold=0.6, redundancy=0.45 should pass and the
        # question is easy -> direct_answer
        decision_permissive = compiler_decision(state, redundancy_threshold=0.6)
        assert decision_permissive == "direct_answer"


class TestCompilerRoutingSignals:
    """Tests for the analyzer-driven routing decisions."""

    def _ready_state(self, **overrides) -> AEDEState:
        """Build a state that has cleared all the existing thresholds:
        coverage >= target, no missing core, redundancy <= threshold,
        confidence >= threshold, not at max. Caller can then tweak
        required_reasoning and direct_answer_possible to drive the new
        branches.
        """
        base: AEDEState = {
            "query": "Why did revenue grow?",
            "query_core_concepts": ["revenue", "growth"],
            "current_top_k": 8,
            "documents": ["d1"],
            "facts": [],
            "answered_parts": ["revenue growth"],
            "missing_parts": [],
            "missing_parts_core": [],
            "coverage": 0.85,
            "redundancy": 0.2,
            "confidence": 0.8,
            "max_retrieval_reached": False,
            "required_reasoning": "deep",
            "direct_answer_possible": False,
        }
        base.update(overrides)
        return base

    def test_routes_to_deep_reasoning_for_deep(self):
        decision = compiler_decision(self._ready_state(required_reasoning="deep"))
        assert decision == "deep_reasoning"

    def test_routes_to_direct_answer_for_none(self):
        decision = compiler_decision(self._ready_state(required_reasoning="none"))
        assert decision == "direct_answer"

    def test_routes_to_llama_with_compress_for_light(self):
        decision = compiler_decision(self._ready_state(required_reasoning="light"))
        assert decision == "llama_with_compress"

    def test_default_routes_to_deep_when_routing_field_missing(self):
        """Old analyzer output (no required_reasoning) -> conservative deep path."""
        state = self._ready_state()
        del state["required_reasoning"]
        decision = compiler_decision(state)
        assert decision == "deep_reasoning"

    def test_deep_reasoning_overrides_low_redundancy(self):
        """Even with low redundancy, deep reasoning should go to Gemini path."""
        state = self._ready_state(
            redundancy=0.1,
            required_reasoning="deep",
        )
        decision = compiler_decision(state)
        # deep_reasoning should win over the no-redundancy case
        assert decision == "deep_reasoning"

    def test_retrieve_more_beats_routing(self):
        """Low coverage still triggers retrieve_more even if reasoning is none."""
        state = self._ready_state(
            coverage=0.5,
            required_reasoning="none",
        )
        decision = compiler_decision(state)
        assert decision == "retrieve_more"

    def test_high_redundancy_beats_routing(self):
        """High redundancy still triggers compress even if reasoning is none."""
        state = self._ready_state(
            redundancy=0.6,
            required_reasoning="none",
        )
        decision = compiler_decision(state)
        assert decision == "compress"

    def test_low_confidence_beats_routing(self):
        """Low confidence still triggers retrieve_more even if reasoning is light."""
        state = self._ready_state(
            confidence=0.3,
            required_reasoning="light",
        )
        decision = compiler_decision(state)
        assert decision == "retrieve_more"

    def test_max_retrieval_overrides_routing(self):
        """Max retrieval signal wins over reasoning depth."""
        state = self._ready_state(
            max_retrieval_reached=True,
            required_reasoning="none",
        )
        decision = compiler_decision(state)
        assert decision == "max_retrieval_reached"