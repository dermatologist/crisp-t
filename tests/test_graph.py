"""
Tests for graph.py
"""

import pytest
import pandas as pd

from src.crisp_t.graph import CrispGraph
from src.crisp_t.model.corpus import Corpus
from src.crisp_t.model.document import Document


@pytest.fixture
def corpus_with_keywords():
    """Create a corpus with documents that have keywords."""
    doc1 = Document(
        id="doc1",
        name="Document 1",
        text="This is document one",
        metadata={"keywords": ["health", "research"]}
    )
    doc2 = Document(
        id="doc2",
        name="Document 2",
        text="This is document two",
        metadata={"keywords": ["health", "policy"]}
    )
    doc3 = Document(
        id="doc3",
        name="Document 3",
        text="This is document three",
        metadata={"keywords": "research, policy, education"}
    )
    
    corpus = Corpus(
        id="test_corpus",
        name="Test Corpus",
        documents=[doc1, doc2, doc3]
    )
    return corpus


@pytest.fixture
def corpus_without_keywords():
    """Create a corpus without keywords."""
    doc1 = Document(
        id="doc1",
        name="Document 1",
        text="This is document one"
    )
    doc2 = Document(
        id="doc2",
        name="Document 2",
        text="This is document two"
    )
    
    corpus = Corpus(
        id="test_corpus",
        name="Test Corpus",
        documents=[doc1, doc2]
    )
    return corpus


@pytest.fixture
def corpus_with_clusters():
    """Create a corpus with cluster information."""
    doc1 = Document(
        id="doc1",
        name="Document 1",
        text="This is document one",
        metadata={"keywords": ["health"], "cluster": 0}
    )
    doc2 = Document(
        id="doc2",
        name="Document 2",
        text="This is document two",
        metadata={"keywords": ["policy"], "cluster": 1}
    )
    
    corpus = Corpus(
        id="test_corpus",
        name="Test Corpus",
        documents=[doc1, doc2],
        metadata={"clusters": {0: {"name": "Cluster 0"}, 1: {"name": "Cluster 1"}}}
    )
    return corpus


@pytest.fixture
def corpus_with_df():
    """Create a corpus with DataFrame metadata."""
    doc1 = Document(
        id="doc1",
        name="Document 1",
        text="This is document one",
        metadata={"keywords": ["health"]}
    )
    doc2 = Document(
        id="doc2",
        name="Document 2",
        text="This is document two",
        metadata={"keywords": ["policy"]}
    )
    
    df = pd.DataFrame({
        "id": ["doc1", "doc2"],
        "score": [0.8, 0.9],
        "category": ["A", "B"]
    })
    
    corpus = Corpus(
        id="test_corpus",
        name="Test Corpus",
        documents=[doc1, doc2],
        df=df
    )
    return corpus


def test_crisp_graph_init(corpus_with_keywords):
    """Test CrispGraph initialization."""
    graph = CrispGraph(corpus_with_keywords)
    assert graph.corpus == corpus_with_keywords
    assert graph.graph is None


def test_create_graph_basic(corpus_with_keywords):
    """Test basic graph creation with keywords."""
    graph = CrispGraph(corpus_with_keywords)
    graph_data = graph.create_graph()
    
    assert "nodes" in graph_data
    assert "edges" in graph_data
    assert graph_data["num_nodes"] > 0
    assert graph_data["num_edges"] > 0
    assert graph_data["num_documents"] == 3
    assert graph_data["has_keywords"] is True


def test_create_graph_without_keywords(corpus_without_keywords):
    """Test that graph creation fails without keywords."""
    graph = CrispGraph(corpus_without_keywords)
    
    with pytest.raises(ValueError, match="keywords assigned"):
        graph.create_graph()


def test_graph_stored_in_corpus_metadata(corpus_with_keywords):
    """Test that graph data is stored in corpus metadata."""
    graph = CrispGraph(corpus_with_keywords)
    graph.create_graph()
    
    assert "graph" in corpus_with_keywords.metadata
    assert isinstance(corpus_with_keywords.metadata["graph"], dict)


def test_document_nodes_created(corpus_with_keywords):
    """Test that document nodes are created correctly."""
    graph = CrispGraph(corpus_with_keywords)
    graph_data = graph.create_graph()
    
    doc_nodes = [n for n in graph_data["nodes"] if n["label"] == "document"]
    assert len(doc_nodes) == 3
    
    # Check that doc IDs match
    doc_ids = [n["id"] for n in doc_nodes]
    assert "doc1" in doc_ids
    assert "doc2" in doc_ids
    assert "doc3" in doc_ids


def test_keyword_nodes_created(corpus_with_keywords):
    """Test that keyword nodes are created correctly."""
    graph = CrispGraph(corpus_with_keywords)
    graph_data = graph.create_graph()
    
    keyword_nodes = [n for n in graph_data["nodes"] if n["label"] == "keyword"]
    # Should have unique keywords: health, research, policy, education
    assert len(keyword_nodes) == 4
    
    keyword_ids = [n["id"] for n in keyword_nodes]
    assert "keyword:health" in keyword_ids
    assert "keyword:research" in keyword_ids
    assert "keyword:policy" in keyword_ids
    assert "keyword:education" in keyword_ids


def test_document_keyword_edges(corpus_with_keywords):
    """Test that edges connect documents to keywords."""
    graph = CrispGraph(corpus_with_keywords)
    graph_data = graph.create_graph()
    
    edges = graph_data["edges"]
    assert len(edges) > 0
    
    # Check that edges connect documents to keywords
    keyword_edges = [e for e in edges if e["label"] == "HAS_KEYWORD"]
    assert len(keyword_edges) > 0
    
    # Doc1 should connect to health and research
    doc1_edges = [e for e in keyword_edges if e["source"] == "doc1"]
    assert len(doc1_edges) == 2


def test_cluster_nodes_created(corpus_with_clusters):
    """Test that cluster nodes are created when cluster data exists."""
    graph = CrispGraph(corpus_with_clusters)
    graph_data = graph.create_graph()
    
    cluster_nodes = [n for n in graph_data["nodes"] if n["label"] == "cluster"]
    assert len(cluster_nodes) == 2
    
    cluster_ids = [n["id"] for n in cluster_nodes]
    assert "cluster:0" in cluster_ids
    assert "cluster:1" in cluster_ids


def test_cluster_edges_created(corpus_with_clusters):
    """Test that edges connect documents to clusters."""
    graph = CrispGraph(corpus_with_clusters)
    graph_data = graph.create_graph()
    
    cluster_edges = [e for e in graph_data["edges"] if e["label"] == "BELONGS_TO_CLUSTER"]
    assert len(cluster_edges) == 2


def test_metadata_nodes_with_df(corpus_with_df):
    """Test that metadata nodes are created from DataFrame."""
    graph = CrispGraph(corpus_with_df)
    graph_data = graph.create_graph()
    
    metadata_nodes = [n for n in graph_data["nodes"] if n["label"] == "metadata"]
    assert len(metadata_nodes) == 2
    
    # Check properties
    for node in metadata_nodes:
        assert "score" in node["properties"]
        assert "category" in node["properties"]


def test_metadata_edges_with_df(corpus_with_df):
    """Test that edges connect documents to metadata."""
    graph = CrispGraph(corpus_with_df)
    graph_data = graph.create_graph()
    
    metadata_edges = [e for e in graph_data["edges"] if e["label"] == "HAS_METADATA"]
    assert len(metadata_edges) == 2


def test_get_networkx_graph(corpus_with_keywords):
    """Test getting NetworkX graph representation."""
    graph = CrispGraph(corpus_with_keywords)
    graph.create_graph()
    
    nx_graph = graph.get_networkx_graph()
    assert nx_graph is not None
    assert nx_graph.number_of_nodes() > 0
    assert nx_graph.number_of_edges() > 0


def test_get_networkx_graph_before_creation(corpus_with_keywords):
    """Test that getting graph before creation raises error."""
    graph = CrispGraph(corpus_with_keywords)
    
    with pytest.raises(ValueError, match="Graph not created yet"):
        graph.get_networkx_graph()


def test_corpus_without_df_no_id_field():
    """Test that corpus without aligning ID field logs warning."""
    doc1 = Document(
        id="doc1",
        name="Document 1",
        text="This is document one",
        metadata={"keywords": ["health"]}
    )
    
    # DataFrame without ID field
    df = pd.DataFrame({
        "score": [0.8],
        "category": ["A"]
    })
    
    corpus = Corpus(
        id="test_corpus",
        name="Test Corpus",
        documents=[doc1],
        df=df
    )
    
    graph = CrispGraph(corpus)
    graph_data = graph.create_graph()
    
    # Should create graph without metadata nodes
    metadata_nodes = [n for n in graph_data["nodes"] if n["label"] == "metadata"]
    assert len(metadata_nodes) == 0
    assert graph_data["has_metadata"] is False
