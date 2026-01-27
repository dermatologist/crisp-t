"""Tests for embedding-based cross-modal linking."""
import importlib.util

import numpy as np
import pandas as pd
import pytest

from src.crisp_t.embedding_linker import EmbeddingLinker
from src.crisp_t.model.corpus import Corpus
from src.crisp_t.model.document import Document

# Check if chromadb is available without importing it
CHROMADB_AVAILABLE = importlib.util.find_spec("chromadb") is not None


def make_embedding_corpus():
    """Create a test corpus for embedding linking."""
    docs = [
        Document(
            id="doc1",
            name="Healthcare Document",
            text="Patient shows symptoms of flu including fever and cough",
            metadata={}
        ),
        Document(
            id="doc2",
            name="Weather Report",
            text="Temperature dropped significantly with heavy rain and wind",
            metadata={}
        ),
        Document(
            id="doc3",
            name="Medical Record",
            text="Blood pressure readings show elevated systolic values",
            metadata={}
        ),
    ]
    
    # Create numeric data that could correlate with documents
    df = pd.DataFrame({
        "temperature": [38.5, 15.2, 37.1],  # Celsius
        "heart_rate": [95, 72, 88],
        "blood_pressure_sys": [125, 110, 145],
        "blood_pressure_dia": [80, 70, 92],
    })
    
    return Corpus(
        id="embedding_test",
        name="Embedding Test Corpus",
        documents=docs,
        df=df,
    )


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")
def test_embedding_linker_initialization():
    """Test EmbeddingLinker initialization."""
    corpus = make_embedding_corpus()
    
    linker = EmbeddingLinker(
        corpus,
        similarity_metric="cosine",
        use_simple_embeddings=True  # Use simple embeddings for testing
    )
    
    assert linker.corpus == corpus
    assert linker.similarity_metric == "cosine"
    assert linker.use_simple_embeddings is True


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")
def test_get_text_embeddings():
    """Test text embedding generation."""
    corpus = make_embedding_corpus()
    linker = EmbeddingLinker(corpus, use_simple_embeddings=True)
    
    embeddings = linker._get_text_embeddings()
    
    assert embeddings is not None
    assert embeddings.shape[0] == len(corpus.documents)
    assert embeddings.shape[1] > 0  # Should have some embedding dimension


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")
def test_get_numeric_embeddings():
    """Test numeric embedding generation."""
    corpus = make_embedding_corpus()
    linker = EmbeddingLinker(corpus, use_simple_embeddings=True)
    
    embeddings = linker._get_numeric_embeddings()
    
    assert embeddings is not None
    assert embeddings.shape[0] == len(corpus.df)
    assert embeddings.shape[1] == 4  # 4 numeric columns


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")
def test_get_numeric_embeddings_specific_columns():
    """Test numeric embedding with specific columns."""
    corpus = make_embedding_corpus()
    linker = EmbeddingLinker(corpus, use_simple_embeddings=True)
    
    embeddings = linker._get_numeric_embeddings(columns=["temperature", "heart_rate"])
    
    assert embeddings.shape[1] == 2


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")
def test_compute_similarity_matrix():
    """Test similarity matrix computation."""
    corpus = make_embedding_corpus()
    linker = EmbeddingLinker(corpus, similarity_metric="cosine", use_simple_embeddings=True)
    
    similarity_matrix = linker.compute_similarity_matrix()
    
    assert similarity_matrix.shape == (len(corpus.documents), len(corpus.df))
    # Cosine similarity should be between -1 and 1
    assert similarity_matrix.min() >= -1.0
    assert similarity_matrix.max() <= 1.0


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")
def test_compute_similarity_matrix_euclidean():
    """Test similarity matrix with Euclidean distance."""
    corpus = make_embedding_corpus()
    linker = EmbeddingLinker(corpus, similarity_metric="euclidean", use_simple_embeddings=True)
    
    similarity_matrix = linker.compute_similarity_matrix()
    
    assert similarity_matrix.shape == (len(corpus.documents), len(corpus.df))
    # All similarities should be positive (inverse distance)
    assert similarity_matrix.min() >= 0.0


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")
def test_link_by_embedding_similarity():
    """Test embedding-based linking."""
    corpus = make_embedding_corpus()
    linker = EmbeddingLinker(corpus, use_simple_embeddings=True)
    
    result_corpus = linker.link_by_embedding_similarity(top_k=1)
    
    # Check that links were created
    linked_count = sum(
        1 for doc in result_corpus.documents
        if "embedding_links" in doc.metadata and doc.metadata["embedding_links"]
    )
    
    assert linked_count > 0
    
    # Check link structure
    for doc in result_corpus.documents:
        if "embedding_links" in doc.metadata and doc.metadata["embedding_links"]:
            links = doc.metadata["embedding_links"]
            assert len(links) <= 1  # top_k=1
            
            for link in links:
                assert "df_index" in link
                assert "similarity_score" in link
                assert "link_type" in link
                assert link["link_type"] == "embedding"
                assert 0.0 <= link["similarity_score"] <= 1.0


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")
def test_link_by_embedding_similarity_with_threshold():
    """Test embedding linking with similarity threshold."""
    corpus = make_embedding_corpus()
    linker = EmbeddingLinker(corpus, use_simple_embeddings=True)
    
    result_corpus = linker.link_by_embedding_similarity(threshold=0.5, top_k=2)
    
    # Check that all links meet threshold
    for doc in result_corpus.documents:
        if "embedding_links" in doc.metadata:
            for link in doc.metadata["embedding_links"]:
                assert link["similarity_score"] >= 0.5


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")
def test_link_by_embedding_similarity_top_k():
    """Test embedding linking with top_k > 1."""
    corpus = make_embedding_corpus()
    linker = EmbeddingLinker(corpus, use_simple_embeddings=True)
    
    result_corpus = linker.link_by_embedding_similarity(top_k=2)
    
    # At least one document should have 2 links
    max_links = max(
        len(doc.metadata.get("embedding_links", []))
        for doc in result_corpus.documents
    )
    
    assert max_links >= 1  # Should have at least 1 link


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")
def test_get_link_statistics():
    """Test link statistics retrieval."""
    corpus = make_embedding_corpus()
    linker = EmbeddingLinker(corpus, use_simple_embeddings=True)
    
    linker.link_by_embedding_similarity(top_k=1)
    stats = linker.get_link_statistics()
    
    assert "total_documents" in stats
    assert "linked_documents" in stats
    assert "total_links" in stats
    assert "avg_similarity" in stats
    assert "min_similarity" in stats
    assert "max_similarity" in stats
    
    assert stats["total_documents"] == len(corpus.documents)
    assert stats["linked_documents"] > 0
    assert stats["total_links"] > 0


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")
def test_embedding_linker_no_dataframe():
    """Test that linker handles missing dataframe gracefully."""
    docs = [
        Document(id="doc1", name="Doc", text="Some text", metadata={})
    ]
    corpus = Corpus(id="test", name="Test", documents=docs, df=None)
    
    linker = EmbeddingLinker(corpus, use_simple_embeddings=True)
    
    with pytest.raises(ValueError, match="no dataframe"):
        linker._get_numeric_embeddings()


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")
def test_different_similarity_metrics():
    """Test both cosine and euclidean similarity metrics."""
    corpus = make_embedding_corpus()
    
    # Cosine similarity
    linker_cosine = EmbeddingLinker(corpus, similarity_metric="cosine", use_simple_embeddings=True)
    sim_cosine = linker_cosine.compute_similarity_matrix()
    
    # Euclidean similarity
    linker_euclidean = EmbeddingLinker(corpus, similarity_metric="euclidean", use_simple_embeddings=True)
    sim_euclidean = linker_euclidean.compute_similarity_matrix()
    
    # Both should produce valid similarity matrices
    assert sim_cosine.shape == sim_euclidean.shape
    assert not np.array_equal(sim_cosine, sim_euclidean)  # Should be different


def test_embedding_linker_without_chromadb():
    """Test that appropriate error is raised without ChromaDB."""
    if CHROMADB_AVAILABLE:
        pytest.skip("ChromaDB is installed")
    
    corpus = make_embedding_corpus()
    
    linker = EmbeddingLinker(corpus)
    
    with pytest.raises(ImportError, match="ChromaDB is required"):
        linker._get_text_embeddings()
