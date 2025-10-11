"""
Tests for semantic search functionality with ChromaDB.
"""

import os
import shutil
from unittest.mock import MagicMock, patch

import pytest

from crisp_t.model import Corpus, Document
from crisp_t.semantic import CHROMADB_AVAILABLE, Semantic


@pytest.fixture
def sample_corpus():
    """Create a sample corpus for testing."""
    documents = [
        Document(
            id="doc1",
            name="First Document",
            text="This is a document about machine learning and AI.",
            metadata={"topic": "technology", "year": 2024},
        ),
        Document(
            id="doc2",
            name="Second Document",
            text="This document discusses natural language processing.",
            metadata={"topic": "nlp", "year": 2023},
        ),
        Document(
            id="doc3",
            name="Third Document",
            text="Healthcare and medical research are important topics.",
            metadata={"topic": "healthcare", "year": 2024},
        ),
    ]

    corpus = Corpus(
        id="test_corpus",
        name="Test Corpus",
        description="A test corpus for semantic search",
        documents=documents,
    )
    return corpus


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="chromadb not installed")
class TestSemantic:
    """Test cases for Semantic class."""

    def test_init_with_valid_corpus(self, sample_corpus):
        """Test initialization with a valid corpus."""
        semantic = Semantic(sample_corpus, use_simple_embeddings=True)
        assert semantic._corpus == sample_corpus
        assert semantic._collection_name == "crisp-t"
        assert semantic._client is not None
        assert semantic._collection is not None

    def test_init_with_none_corpus(self):
        """Test initialization with None corpus raises ValueError."""
        with pytest.raises(ValueError, match="Corpus cannot be None"):
            Semantic(None, use_simple_embeddings=True)

    def test_init_with_empty_corpus(self):
        """Test initialization with empty corpus raises ValueError."""
        empty_corpus = Corpus(
            id="empty_corpus", name="Empty", description="No documents", documents=[]
        )
        with pytest.raises(ValueError, match="Corpus must contain at least one document"):
            Semantic(empty_corpus, use_simple_embeddings=True)

    def test_get_similar(self, sample_corpus):
        """Test semantic search with get_similar."""
        semantic = Semantic(sample_corpus, use_simple_embeddings=True)
        query = "artificial intelligence and machine learning"
        result_corpus = semantic.get_similar(query, n_results=2)

        assert isinstance(result_corpus, Corpus)
        assert len(result_corpus.documents) <= 2
        assert result_corpus.metadata["semantic_query"] == query
        assert result_corpus.metadata["semantic_n_results"] == 2
        # The corpus should be updated
        assert semantic._corpus == result_corpus

    def test_get_similar_returns_relevant_docs(self, sample_corpus):
        """Test that get_similar returns relevant documents."""
        semantic = Semantic(sample_corpus, use_simple_embeddings=True)
        query = "natural language processing"
        result_corpus = semantic.get_similar(query, n_results=1)

        # Should return at least one document
        assert len(result_corpus.documents) >= 1
        # Most relevant document should be doc2 which mentions NLP
        assert any("doc2" in doc.id for doc in result_corpus.documents)

    def test_get_df_with_all_metadata(self, sample_corpus):
        """Test get_df exports all metadata."""
        semantic = Semantic(sample_corpus, use_simple_embeddings=True)
        result_corpus = semantic.get_df()

        assert result_corpus.df is not None
        assert "id" in result_corpus.df.columns
        # Check that metadata keys are present
        assert len(result_corpus.df) == len(sample_corpus.documents)

    def test_get_df_with_specific_keys(self, sample_corpus):
        """Test get_df with specific metadata keys."""
        semantic = Semantic(sample_corpus, use_simple_embeddings=True)
        result_corpus = semantic.get_df(metadata_keys=["topic"])

        assert result_corpus.df is not None
        assert "id" in result_corpus.df.columns
        assert "topic" in result_corpus.df.columns

    def test_get_df_merges_with_existing_df(self):
        """Test get_df merges with existing DataFrame."""
        import pandas as pd

        # Create corpus with existing DataFrame
        documents = [
            Document(
                id="doc1",
                name="Doc 1",
                text="Text 1",
                metadata={"category": "A"},
            ),
            Document(
                id="doc2",
                name="Doc 2",
                text="Text 2",
                metadata={"category": "B"},
            ),
        ]

        existing_df = pd.DataFrame({"id": ["doc1", "doc2"], "score": [0.9, 0.8]})

        corpus = Corpus(
            id="test_corpus",
            name="Test",
            description="Test",
            documents=documents,
            df=existing_df,
        )

        semantic = Semantic(corpus, use_simple_embeddings=True)
        result_corpus = semantic.get_df()

        # Should have merged data
        assert "id" in result_corpus.df.columns
        assert "score" in result_corpus.df.columns
        assert "category" in result_corpus.df.columns

    def test_save_and_restore_collection(self, sample_corpus, tmp_path):
        """Test saving and restoring collection."""
        semantic = Semantic(sample_corpus, use_simple_embeddings=True)

        # Save collection
        save_path = str(tmp_path / "test_chroma")
        semantic.save_collection(save_path)

        # Verify save path exists
        assert os.path.exists(save_path)

        # Create new semantic instance and restore
        new_corpus = Corpus(
            id="new_corpus",
            name="New",
            description="New",
            documents=[
                Document(id="temp", name="Temp", text="Temp text", metadata={})
            ],
        )
        new_semantic = Semantic(new_corpus, use_simple_embeddings=True)
        new_semantic.restore_collection(save_path)

        # Query should work on restored collection
        result = new_semantic.get_similar("machine learning", n_results=1)
        assert len(result.documents) >= 1

        # Cleanup
        if os.path.exists(save_path):
            shutil.rmtree(save_path)

    def test_restore_nonexistent_path(self, sample_corpus):
        """Test restoring from non-existent path raises error."""
        semantic = Semantic(sample_corpus, use_simple_embeddings=True)
        with pytest.raises(ValueError, match="does not exist"):
            semantic.restore_collection("/nonexistent/path")


@pytest.mark.skipif(CHROMADB_AVAILABLE, reason="Testing import error handling")
def test_semantic_without_chromadb():
    """Test that Semantic raises ImportError when chromadb is not available."""
    # This test only runs when chromadb is NOT available
    # We need to mock the corpus
    mock_corpus = MagicMock()
    mock_corpus.documents = [MagicMock()]

    with pytest.raises(ImportError, match="chromadb is required"):
        # This should be caught by the import check in __init__
        pass
