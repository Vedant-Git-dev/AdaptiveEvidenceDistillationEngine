"""Tests for focused_retriever module."""

from unittest.mock import patch, MagicMock

import pytest

from aede.state import AEDEState, create_initial_state
from aede.nodes.retrieval import focused_retriever, add_documents, MockCollection


class TestFocusedRetriever:
    """Test suite for focused_retriever function."""

    def test_retriever_returns_documents(self, mock_urllib_response=None):
        """Test that retriever returns documents from vector store."""
        with patch("aede.nodes.retrieval.get_or_create_vectorstore") as mock_get:
            mock_collection = MockCollection()
            mock_get.return_value = mock_collection

            state = create_initial_state("Test query")
            result = focused_retriever(state)

            assert "documents" in result
            assert "current_top_k" in result
            assert result["current_top_k"] == 4
            assert isinstance(result["documents"], list)

    def test_retriever_with_custom_k(self):
        """Test retriever uses custom k value from state."""
        with patch("aede.nodes.retrieval.get_or_create_vectorstore") as mock_get:
            mock_collection = MockCollection()
            mock_get.return_value = mock_collection

            state = create_initial_state("Test query")
            state["current_top_k"] = 8

            result = focused_retriever(state)

            assert result["current_top_k"] == 8
            assert len(mock_collection.query_calls) == 1
            assert mock_collection.query_calls[0]["n_results"] == 8

    def test_retriever_updates_workflow_path(self):
        """Test that workflow path is updated."""
        with patch("aede.nodes.retrieval.get_or_create_vectorstore") as mock_get:
            mock_get.return_value = MockCollection()

            state = create_initial_state("Test query")
            state["workflow_path"] = ["start"]

            result = focused_retriever(state)

            assert "focused_retriever" in result["workflow_path"]

    def test_retriever_handles_empty_state(self):
        """Test retriever handles minimal state."""
        with patch("aede.nodes.retrieval.get_or_create_vectorstore") as mock_get:
            mock_get.return_value = MockCollection()

            state: AEDEState = {
                "query": "Test",
                "current_top_k": 4,
            }  # type: ignore

            result = focused_retriever(state)

            assert result["current_top_k"] == 4
            assert "documents" in result


class TestAddDocuments:
    """Test suite for add_documents function."""

    def test_add_documents_calls_collection_add(self):
        """Test that add_documents adds documents to vector store."""
        import numpy as np
        with patch("aede.nodes.retrieval.get_or_create_vectorstore") as mock_get:
            with patch("sentence_transformers.SentenceTransformer") as mock_st:
                mock_collection = MagicMock()
                mock_get.return_value = mock_collection
                mock_st.return_value.encode.return_value = np.array([[0.1] * 1024] * 3)

                docs = ["Doc 1", "Doc 2", "Doc 3"]
                result = add_documents(docs)

                assert result == 3
                mock_collection.add.assert_called_once()

    def test_add_documents_with_custom_ids(self):
        """Test add_documents with custom document IDs."""
        import numpy as np
        with patch("aede.nodes.retrieval.get_or_create_vectorstore") as mock_get:
            with patch("sentence_transformers.SentenceTransformer") as mock_st:
                mock_collection = MagicMock()
                mock_get.return_value = mock_collection
                mock_st.return_value.encode.return_value = np.array([[0.1] * 1024])

                docs = ["Single doc"]
                ids = ["custom_id_1"]
                add_documents(docs, ids=ids)

                call_kwargs = mock_collection.add.call_args.kwargs
                assert call_kwargs["ids"] == ids


class TestMockCollection:
    """Test suite for MockCollection used in testing."""

    def test_query_returns_requested_count(self):
        """Test that query returns up to n_results documents."""
        collection = MockCollection(SAMPLE_DOCUMENTS)

        result = result = collection.query(
            query_texts=["test"],
            n_results=2,
        )

        docs = result["documents"][0]
        assert len(docs) == 2

    def test_query_tracks_calls(self):
        """Test that query tracks its calls."""
        collection = MockCollection()

        collection.query(query_texts=["query1"], n_results=4)
        collection.query(query_texts=["query2"], n_results=2)

        assert len(collection.query_calls) == 2
        assert collection.query_calls[0]["n_results"] == 4
        assert collection.query_calls[1]["n_results"] == 2


# Re-export sample documents for use in conftest
SAMPLE_DOCUMENTS = [
    "Revenue grew by 15% in Q3 due to increased sales in the enterprise segment.",
    "Q3 revenue reached $50 million, up from $43 million in Q2.",
    "CEO stated that the company's focus on enterprise customers has been successful.",
    "The company launched three new products in Q3 2023.",
]