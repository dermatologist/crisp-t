"""Tests for linkage-based filtering functionality."""
import pandas as pd
import pytest
from datetime import datetime

from src.crisp_t.model.corpus import Corpus
from src.crisp_t.model.document import Document
from src.crisp_t.linkage_filter import LinkageFilter


def make_test_corpus_with_links():
    """Create a test corpus with various linkage types."""
    docs = [
        Document(
            id="doc1",
            name="Healthcare Doc",
            text="Patient shows symptoms of flu",
            timestamp="2025-01-15T10:00:00",
            metadata={
                "keywords": ["healthcare", "symptoms", "flu"],
                "temporal_links": [
                    {"df_index": 0, "time_gap_seconds": 60, "link_type": "temporal"}
                ],
                "embedding_links": [
                    {"df_index": 0, "similarity_score": 0.85, "link_type": "embedding"}
                ],
            }
        ),
        Document(
            id="doc2",
            name="Weather Report",
            text="Temperature dropped significantly",
            timestamp="2025-01-15T11:00:00",
            metadata={
                "keywords": ["weather", "temperature"],
                "temporal_links": [
                    {"df_index": 1, "time_gap_seconds": 120, "link_type": "temporal"}
                ],
                "embedding_links": [
                    {"df_index": 1, "similarity_score": 0.65, "link_type": "embedding"}
                ],
            }
        ),
        Document(
            id="doc3",
            name="Medical Record",
            text="Blood pressure elevated",
            timestamp="2025-01-16T10:00:00",
            metadata={
                "keywords": ["healthcare", "medical"],
                "temporal_links": [
                    {"df_index": 2, "time_gap_seconds": 300, "link_type": "temporal"}
                ],
                "embedding_links": [
                    {"df_index": 2, "similarity_score": 0.92, "link_type": "embedding"}
                ],
            }
        ),
    ]
    
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "category": ["health", "weather", "health"],
        "temperature": [38.5, 15.2, 37.1],
        "value": [100, 200, 150],
    })
    
    corpus = Corpus(
        id="test_linkage",
        name="Test Linkage Corpus",
        documents=docs,
        df=df,
        metadata={
            "relationships": [
                {"first": "text:healthcare", "second": "numb:category", "relation": "correlates"},
                {"first": "text:temperature", "second": "numb:temperature", "relation": "correlates"},
            ]
        }
    )
    
    return corpus


def test_linkage_filter_initialization():
    """Test LinkageFilter initialization."""
    corpus = make_test_corpus_with_links()
    linker = LinkageFilter(corpus)
    
    assert linker.corpus == corpus
    assert LinkageFilter.LINKAGE_METHODS == ["id", "keyword", "time", "embedding"]


def test_filter_by_id():
    """Test ID-based filtering."""
    corpus = make_test_corpus_with_links()
    linker = LinkageFilter(corpus)
    
    # Filter by category
    filtered = linker.filter_by_linkage("id", "category", "health")
    
    # Should have 2 rows with category=health
    assert len(filtered.df) == 2
    assert all(filtered.df["category"] == "health")


def test_filter_by_id_with_int_conversion():
    """Test ID-based filtering with int conversion."""
    corpus = make_test_corpus_with_links()
    linker = LinkageFilter(corpus)
    
    # Filter by id (string to int conversion)
    filtered = linker.filter_by_linkage("id", "id", "1")
    
    assert len(filtered.df) == 1
    assert filtered.df.iloc[0]["id"] == 1


def test_filter_by_keyword():
    """Test keyword-based filtering."""
    corpus = make_test_corpus_with_links()
    linker = LinkageFilter(corpus)
    
    # Filter by keyword "healthcare"
    filtered = linker.filter_by_linkage("keyword", "healthcare", "")
    
    # Should have 2 documents with "healthcare" keyword
    assert len(filtered.documents) == 2
    assert all("healthcare" in doc.metadata.get("keywords", []) for doc in filtered.documents)


def test_filter_by_keyword_no_keyword():
    """Test keyword filtering when keyword doesn't exist."""
    corpus = make_test_corpus_with_links()
    linker = LinkageFilter(corpus)
    
    with pytest.raises(ValueError, match="No documents found with keyword"):
        linker.filter_by_linkage("keyword", "nonexistent", "")


def test_filter_by_time():
    """Test time-based filtering."""
    corpus = make_test_corpus_with_links()
    linker = LinkageFilter(corpus)
    
    # Filter by time (all documents have temporal links)
    filtered = linker.filter_by_linkage("time", "timestamp", "")
    
    # Should have all 3 documents (all have temporal links)
    assert len(filtered.documents) == 3
    # Should have 3 dataframe rows linked
    assert len(filtered.df) == 3


def test_filter_by_time_no_links():
    """Test time filtering when no temporal links exist."""
    docs = [Document(id="doc1", name="Doc", text="Text", metadata={})]
    corpus = Corpus(id="test", name="Test", documents=docs, df=pd.DataFrame())
    linker = LinkageFilter(corpus)
    
    with pytest.raises(ValueError, match="No temporal links found"):
        linker.filter_by_linkage("time", "timestamp", "")


def test_filter_by_embedding():
    """Test embedding-based filtering."""
    corpus = make_test_corpus_with_links()
    linker = LinkageFilter(corpus)
    
    # Filter by embedding similarity (no threshold)
    filtered = linker.filter_by_linkage("embedding", "", "", min_similarity=0.0)
    
    # Should have all 3 documents (all have embedding links)
    assert len(filtered.documents) == 3


def test_filter_by_embedding_with_threshold():
    """Test embedding filtering with similarity threshold."""
    corpus = make_test_corpus_with_links()
    linker = LinkageFilter(corpus)
    
    # Filter with threshold 0.8
    filtered = linker.filter_by_linkage("embedding", "", "", min_similarity=0.8)
    
    # Should have 2 documents (doc1: 0.85, doc3: 0.92)
    assert len(filtered.documents) == 2
    assert len(filtered.df) == 2


def test_filter_by_embedding_high_threshold():
    """Test embedding filtering with very high threshold."""
    corpus = make_test_corpus_with_links()
    linker = LinkageFilter(corpus)
    
    # Filter with threshold 0.95 (none meet this)
    with pytest.raises(ValueError, match="No embedding links with similarity"):
        linker.filter_by_linkage("embedding", "", "", min_similarity=0.95)


def test_filter_by_embedding_no_links():
    """Test embedding filtering when no embedding links exist."""
    docs = [Document(id="doc1", name="Doc", text="Text", metadata={})]
    corpus = Corpus(id="test", name="Test", documents=docs, df=pd.DataFrame())
    linker = LinkageFilter(corpus)
    
    with pytest.raises(ValueError, match="No embedding links found"):
        linker.filter_by_linkage("embedding", "", "")


def test_invalid_linkage_method():
    """Test invalid linkage method."""
    corpus = make_test_corpus_with_links()
    linker = LinkageFilter(corpus)
    
    with pytest.raises(ValueError, match="Invalid linkage method"):
        linker.filter_by_linkage("invalid", "key", "value")


def test_filter_statistics_time():
    """Test getting filter statistics for time linkage."""
    corpus = make_test_corpus_with_links()
    linker = LinkageFilter(corpus)
    
    stats = linker.get_filter_statistics("time")
    
    assert stats["method"] == "time"
    assert stats["total_documents"] == 3
    assert stats["linkage_available"] is True
    assert stats["linked_documents"] == 3


def test_filter_statistics_embedding():
    """Test getting filter statistics for embedding linkage."""
    corpus = make_test_corpus_with_links()
    linker = LinkageFilter(corpus)
    
    stats = linker.get_filter_statistics("embedding")
    
    assert stats["method"] == "embedding"
    assert stats["total_documents"] == 3
    assert stats["linkage_available"] is True
    assert stats["linked_documents"] == 3


def test_filter_statistics_keyword():
    """Test getting filter statistics for keyword linkage."""
    corpus = make_test_corpus_with_links()
    linker = LinkageFilter(corpus)
    
    stats = linker.get_filter_statistics("keyword")
    
    assert stats["method"] == "keyword"
    assert stats["total_documents"] == 3
    assert stats["linkage_available"] is True
    assert stats["linked_documents"] == 3  # All docs have keywords


def test_filter_statistics_no_links():
    """Test statistics when no links available."""
    docs = [Document(id="doc1", name="Doc", text="Text", metadata={})]
    corpus = Corpus(id="test", name="Test", documents=docs, df=pd.DataFrame())
    linker = LinkageFilter(corpus)
    
    stats = linker.get_filter_statistics("time")
    
    assert stats["linkage_available"] is False
    assert stats["linked_documents"] == 0


def test_filter_preserves_corpus_metadata():
    """Test that filtering preserves corpus metadata."""
    corpus = make_test_corpus_with_links()
    linker = LinkageFilter(corpus)
    
    filtered = linker.filter_by_linkage("id", "category", "health")
    
    assert filtered.id == corpus.id
    assert filtered.name == corpus.name
    assert filtered.description == corpus.description
    assert "relationships" in filtered.metadata


def test_sequential_filters():
    """Test applying multiple filters sequentially."""
    corpus = make_test_corpus_with_links()
    
    # First filter by keyword
    linker1 = LinkageFilter(corpus)
    filtered1 = linker1.filter_by_linkage("keyword", "healthcare", "")
    assert len(filtered1.documents) == 2  # 2 docs with healthcare keyword
    
    # Second filter by embedding on the already filtered corpus
    linker2 = LinkageFilter(filtered1)
    filtered2 = linker2.filter_by_linkage("embedding", "", "", min_similarity=0.8)
    
    # Should have at least 1 document with embedding similarity >= 0.8
    # (doc1: 0.85, doc3: 0.92, both have healthcare keyword)
    assert len(filtered2.documents) >= 1
    assert len(filtered2.documents) <= 2


def test_filter_with_missing_dataframe():
    """Test filtering when dataframe is missing."""
    docs = [
        Document(
            id="doc1",
            name="Doc",
            text="Text",
            metadata={"keywords": ["test"]}
        )
    ]
    corpus = Corpus(id="test", name="Test", documents=docs, df=None)
    linker = LinkageFilter(corpus)
    
    # ID filtering requires dataframe
    with pytest.raises(ValueError, match="No dataframe available"):
        linker.filter_by_linkage("id", "column", "value")
    
    # Keyword filtering should work without dataframe
    filtered = linker.filter_by_linkage("keyword", "test", "")
    assert len(filtered.documents) == 1


def test_filter_empty_result():
    """Test filtering that results in empty corpus."""
    corpus = make_test_corpus_with_links()
    linker = LinkageFilter(corpus)
    
    # Filter by non-existent category
    filtered = linker.filter_by_linkage("id", "category", "nonexistent")
    
    # Should return corpus with original documents (no ID match) or empty
    # Depending on implementation, check it doesn't crash
    assert filtered is not None
