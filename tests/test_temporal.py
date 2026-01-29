"""Tests for temporal analysis functionality."""
from datetime import timedelta

import pandas as pd
import pytest

from src.crisp_t.model.corpus import Corpus
from src.crisp_t.model.document import Document
from src.crisp_t.temporal import TemporalAnalyzer


def make_document_with_timestamp(doc_id, name, timestamp=None):
    """Create a test document with optional timestamp."""
    return Document(
        id=doc_id,
        name=name,
        text=f"Text for {name}",
        timestamp=timestamp,
        metadata={},
    )


def make_temporal_corpus():
    """Create a test corpus with temporal data."""
    docs = [
        make_document_with_timestamp("doc1", "Document 1", "2025-01-15T10:00:00"),
        make_document_with_timestamp("doc2", "Document 2", "2025-01-15T10:30:00"),
        make_document_with_timestamp("doc3", "Document 3", "2025-01-15T11:00:00"),
        make_document_with_timestamp("doc4", "Document 4", None),  # No timestamp
    ]
    df = pd.DataFrame({
        "value": [100, 200, 300],
        "timestamp": ["2025-01-15T10:05:00", "2025-01-15T10:35:00", "2025-01-15T11:10:00"]
    })
    return Corpus(
        id="corpus1",
        name="Test Temporal Corpus",
        documents=docs,
        df=df,
    )


def test_parse_timestamp():
    """Test timestamp parsing."""
    # ISO 8601 format
    dt = TemporalAnalyzer.parse_timestamp("2025-01-15T10:30:00Z")
    assert dt is not None
    assert dt.year == 2025
    assert dt.month == 1
    assert dt.day == 15

    # Different formats
    dt2 = TemporalAnalyzer.parse_timestamp("2025-01-15 10:30:00")
    assert dt2 is not None

    # Invalid timestamp
    dt3 = TemporalAnalyzer.parse_timestamp("invalid")
    assert dt3 is None

    # None input
    dt4 = TemporalAnalyzer.parse_timestamp(None)
    assert dt4 is None


def test_link_by_nearest_time():
    """Test linking documents to nearest dataframe rows."""
    corpus = make_temporal_corpus()
    analyzer = TemporalAnalyzer(corpus)
    
    result = analyzer.link_by_nearest_time(time_column="timestamp")
    
    # Check that documents with timestamps have temporal links
    doc1 = result.get_document_by_id("doc1")
    assert "temporal_links" in doc1.metadata
    assert len(doc1.metadata["temporal_links"]) == 1
    assert doc1.metadata["temporal_links"][0]["link_type"] == "nearest_time"
    
    # Document without timestamp should not have links
    doc4 = result.get_document_by_id("doc4")
    assert "temporal_links" not in doc4.metadata


def test_link_by_time_window():
    """Test linking documents within a time window."""
    corpus = make_temporal_corpus()
    analyzer = TemporalAnalyzer(corpus)
    
    # Window of ±10 minutes
    result = analyzer.link_by_time_window(
        time_column="timestamp",
        window_before=timedelta(minutes=10),
        window_after=timedelta(minutes=10)
    )
    
    # Check that documents have temporal links
    doc1 = result.get_document_by_id("doc1")
    assert "temporal_links" in doc1.metadata
    assert len(doc1.metadata["temporal_links"]) >= 1
    
    # Check link type
    for link in doc1.metadata["temporal_links"]:
        assert link["link_type"] == "time_window"


def test_link_by_sequence():
    """Test sequence-based linking."""
    corpus = make_temporal_corpus()
    analyzer = TemporalAnalyzer(corpus)
    
    # Link by day
    result = analyzer.link_by_sequence(time_column="timestamp", period="D")
    
    # All documents should link to all dataframe rows (same day)
    doc1 = result.get_document_by_id("doc1")
    assert "temporal_links" in doc1.metadata
    assert len(doc1.metadata["temporal_links"]) == 3  # All 3 rows are on same day
    
    # Check link type
    for link in doc1.metadata["temporal_links"]:
        assert link["link_type"] == "sequence"
        assert "period" in link


def test_filter_by_time_range():
    """Test filtering corpus by time range."""
    corpus = make_temporal_corpus()
    analyzer = TemporalAnalyzer(corpus)
    
    # Filter to include only middle document
    result = analyzer.filter_by_time_range(
        start_time="2025-01-15T10:20:00",
        end_time="2025-01-15T10:40:00",
    )
    
    # Should have doc2 and doc4 (no timestamp)
    assert len(result.documents) == 2
    doc_ids = [doc.id for doc in result.documents]
    assert "doc2" in doc_ids
    assert "doc4" in doc_ids  # Documents without timestamps are included
    
    # DataFrame should also be filtered
    assert len(result.df) == 1  # Only middle row


def test_get_temporal_summary():
    """Test temporal summary generation."""
    corpus = make_temporal_corpus()
    analyzer = TemporalAnalyzer(corpus)
    
    # Get daily summary
    summary = analyzer.get_temporal_summary(time_column="timestamp", period="D")
    
    assert not summary.empty
    assert "document_count" in summary.columns or isinstance(summary.columns, pd.MultiIndex)


def test_add_temporal_relationship():
    """Test adding temporal relationships."""
    corpus = make_temporal_corpus()
    analyzer = TemporalAnalyzer(corpus)
    
    # Add temporal relationship
    analyzer.add_temporal_relationship(
        doc_id="doc1",
        df_column="value",
        relation="temporal_correlation"
    )
    
    # Check relationship was added
    rels = corpus.get_relationships()
    assert len(rels) == 1
    assert rels[0]["first"] == "text:doc1"
    assert rels[0]["second"] == "numb:value"
    assert rels[0]["relation"] == "temporal_correlation"


def test_temporal_analyzer_with_no_timestamps():
    """Test temporal analyzer gracefully handles corpus without timestamps."""
    docs = [
        Document(id="doc1", name="Doc 1", text="Text 1"),
        Document(id="doc2", name="Doc 2", text="Text 2"),
    ]
    df = pd.DataFrame({"value": [1, 2, 3]})
    corpus = Corpus(id="corpus1", name="No Timestamps", documents=docs, df=df)
    
    analyzer = TemporalAnalyzer(corpus)
    
    # Should raise error when timestamp column doesn't exist
    with pytest.raises(ValueError):
        analyzer.link_by_nearest_time(time_column="timestamp")


def test_document_timestamp_field():
    """Test that Document model accepts timestamp field."""
    doc = Document(
        id="test1",
        name="Test Doc",
        text="Test text",
        timestamp="2025-01-15T10:00:00Z"
    )
    
    assert doc.timestamp == "2025-01-15T10:00:00Z"
    
    # Test without timestamp (should default to None)
    doc2 = Document(
        id="test2",
        name="Test Doc 2",
        text="Test text 2"
    )
    
    assert doc2.timestamp is None


def test_temporal_sentiment_trend():
    """Test temporal sentiment trend analysis."""
    docs = [
        Document(
            id="doc1",
            name="Doc 1",
            text="Positive text",
            timestamp="2025-01-15T10:00:00",
            metadata={"sentiment": "pos"}
        ),
        Document(
            id="doc2",
            name="Doc 2",
            text="Negative text",
            timestamp="2025-01-15T11:00:00",
            metadata={"sentiment": "neg"}
        ),
        Document(
            id="doc3",
            name="Doc 3",
            text="Neutral text",
            timestamp="2025-01-16T10:00:00",
            metadata={"sentiment": "neu"}
        ),
    ]
    corpus = Corpus(id="corpus1", name="Sentiment Corpus", documents=docs)
    analyzer = TemporalAnalyzer(corpus)
    
    # Get daily sentiment trend
    trend = analyzer.get_temporal_sentiment_trend(period="D")
    
    assert not trend.empty
    assert "sentiment_score" in trend.columns
    assert "document_count" in trend.columns


def test_temporal_topics():
    """Test temporal topic extraction."""
    docs = [
        Document(
            id="doc1",
            name="Doc 1",
            text="machine learning artificial intelligence",
            timestamp="2025-01-15T10:00:00",
            metadata={"topics": ["machine_learning", "AI"]}
        ),
        Document(
            id="doc2",
            name="Doc 2",
            text="deep learning neural networks",
            timestamp="2025-01-15T11:00:00",
            metadata={"topics": ["deep_learning", "neural_networks"]}
        ),
        Document(
            id="doc3",
            name="Doc 3",
            text="data science analytics",
            timestamp="2025-01-16T10:00:00",
            metadata={"topics": ["data_science", "analytics"]}
        ),
    ]
    corpus = Corpus(id="corpus1", name="Topic Corpus", documents=docs)
    analyzer = TemporalAnalyzer(corpus)
    
    # Get weekly topics
    topics = analyzer.get_temporal_topics(period="W", top_n=3)
    
    assert isinstance(topics, dict)
    assert len(topics) > 0
    # Each period should have a list of topics
    for period, topic_list in topics.items():
        assert isinstance(topic_list, list)


def test_temporal_topics_without_metadata():
    """Test temporal topic extraction without topic metadata."""
    docs = [
        Document(
            id="doc1",
            name="Doc 1",
            text="machine learning artificial intelligence technology innovation",
            timestamp="2025-01-15T10:00:00",
        ),
        Document(
            id="doc2",
            name="Doc 2",
            text="deep learning neural networks machine learning",
            timestamp="2025-01-15T11:00:00",
        ),
    ]
    corpus = Corpus(id="corpus1", name="Topic Corpus", documents=docs)
    analyzer = TemporalAnalyzer(corpus)
    
    # Get weekly topics (should fall back to keyword extraction)
    topics = analyzer.get_temporal_topics(period="W", top_n=3)
    
    assert isinstance(topics, dict)
    assert len(topics) > 0
