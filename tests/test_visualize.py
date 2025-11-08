import matplotlib

matplotlib.use("Agg")

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import Rectangle

from src.crisp_t.visualize import QRVisualize, PYLDAVIS_AVAILABLE


@pytest.fixture
def visualize() -> QRVisualize:
    return QRVisualize()


def test_plot_frequency_distribution_of_words_returns_figure(visualize: QRVisualize):
    df = pd.DataFrame({"Text": ["hello world", "test", "another document"]})

    fig, ax = visualize.plot_frequency_distribution_of_words(df, show=False)

    assert fig is not None
    assert ax is not None
    assert len(ax.patches) > 0
    plt.close(fig)


def test_plot_distribution_by_topic_axes_shape(visualize: QRVisualize):
    df = pd.DataFrame(
        {
            "Dominant_Topic": [0, 0, 1, 1, 2, 2],
            "Text": ["a", "bb", "ccc", "dddd", "ee", "fff"],
        }
    )

    fig, axes = visualize.plot_distribution_by_topic(df, show=False, bins=10)

    assert fig is not None
    assert axes.shape[0] * axes.shape[1] >= 3
    plt.close(fig)


def test_plot_top_terms_validates_top_n(visualize: QRVisualize):
    df = pd.DataFrame({"term": ["a"], "frequency": [10]})

    with pytest.raises(ValueError):
        visualize.plot_top_terms(df, top_n=0)


def test_plot_top_terms_generates_bar_chart(visualize: QRVisualize):
    df = pd.DataFrame(
        {
            "term": ["a", "b", "c", "d"],
            "frequency": [10, 20, 5, 15],
        }
    )

    fig, ax = visualize.plot_top_terms(df, top_n=3, show=False)

    assert len(ax.patches) == 3
    widths = [cast(Rectangle, patch).get_width() for patch in ax.patches]
    assert widths == sorted(widths)
    plt.close(fig)


def test_plot_correlation_heatmap_requires_two_numeric_columns(visualize: QRVisualize):
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [4, 3, 2, 1],
            "c": ["x", "y", "z", "w"],
        }
    )

    fig, ax = visualize.plot_correlation_heatmap(df, columns=["a", "b"], show=False)

    assert fig is not None
    assert ax is not None
    assert ax.collections
    quadmesh = ax.collections[0]
    data = np.asarray(quadmesh.get_array()).reshape(2, 2)
    assert np.allclose(data, np.array([[1, -1], [-1, 1]]))
    plt.close(fig)


def test_plot_correlation_heatmap_raises_for_insufficient_numeric_columns(
    visualize: QRVisualize,
) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    with pytest.raises(ValueError):
        visualize.plot_correlation_heatmap(df, columns=["b"], show=False)


def test_get_lda_viz_raises_without_pyldavis(visualize: QRVisualize):
    """Test that get_lda_viz raises ImportError when pyLDAvis is not available"""
    if not PYLDAVIS_AVAILABLE:
        with pytest.raises(ImportError, match="pyLDAvis is not installed"):
            visualize.get_lda_viz(None, None, None)


def test_get_lda_viz_raises_without_lda_model(visualize: QRVisualize):
    """Test that get_lda_viz raises ValueError when LDA model is None"""
    if PYLDAVIS_AVAILABLE:
        with pytest.raises(ValueError, match="LDA model is required"):
            visualize.get_lda_viz(None, [], {})


def test_get_lda_viz_raises_without_corpus_bow(visualize: QRVisualize):
    """Test that get_lda_viz raises ValueError when corpus_bow is None"""
    if PYLDAVIS_AVAILABLE:
        # Create a mock LDA model
        mock_lda = type('MockLDA', (), {})()
        with pytest.raises(ValueError, match="Corpus bag of words is required"):
            visualize.get_lda_viz(mock_lda, None, {})


def test_get_lda_viz_raises_without_dictionary(visualize: QRVisualize):
    """Test that get_lda_viz raises ValueError when dictionary is None"""
    if PYLDAVIS_AVAILABLE:
        # Create a mock LDA model
        mock_lda = type('MockLDA', (), {})()
        with pytest.raises(ValueError, match="Dictionary is required"):
            visualize.get_lda_viz(mock_lda, [], None)


def test_draw_tdabm_basic(visualize: QRVisualize):
    """Test basic TDABM visualization"""
    from src.crisp_t.model.corpus import Corpus
    
    # Create test corpus with TDABM metadata
    corpus = Corpus(
        id="test",
        name="Test",
        metadata={
            'tdabm': {
                'y_variable': 'y',
                'x_variables': ['x1', 'x2'],
                'radius': 0.3,
                'num_landmarks': 3,
                'num_points': 10,
                'landmarks': [
                    {
                        'id': 'B0',
                        'location': [0.2, 0.3],
                        'point_indices': [0, 1, 2],
                        'count': 3,
                        'mean_y': 0.5,
                        'connections': ['B1']
                    },
                    {
                        'id': 'B1',
                        'location': [0.5, 0.6],
                        'point_indices': [3, 4, 5],
                        'count': 3,
                        'mean_y': 0.7,
                        'connections': ['B0', 'B2']
                    },
                    {
                        'id': 'B2',
                        'location': [0.8, 0.4],
                        'point_indices': [6, 7, 8, 9],
                        'count': 4,
                        'mean_y': 0.3,
                        'connections': ['B1']
                    }
                ]
            }
        }
    )
    
    fig = visualize.draw_tdabm(corpus=corpus, show=False)
    
    assert fig is not None
    plt.close(fig)


def test_draw_tdabm_without_metadata(visualize: QRVisualize):
    """Test TDABM visualization fails without metadata"""
    from src.crisp_t.model.corpus import Corpus
    
    corpus = Corpus(id="test", name="Test")
    
    with pytest.raises(ValueError, match="does not contain 'tdabm' data"):
        visualize.draw_tdabm(corpus=corpus, show=False)


def test_draw_tdabm_without_corpus(visualize: QRVisualize):
    """Test TDABM visualization fails without corpus"""
    
    with pytest.raises(ValueError, match="No corpus provided"):
        visualize.draw_tdabm(corpus=None, show=False)


def test_draw_tdabm_1d_coordinates(visualize: QRVisualize):
    """Test TDABM visualization with 1D coordinates"""
    from src.crisp_t.model.corpus import Corpus
    
    # Create test corpus with 1D landmarks
    corpus = Corpus(
        id="test",
        name="Test",
        metadata={
            'tdabm': {
                'y_variable': 'y',
                'x_variables': ['x1'],
                'radius': 0.3,
                'num_landmarks': 2,
                'num_points': 5,
                'landmarks': [
                    {
                        'id': 'B0',
                        'location': [0.2],
                        'point_indices': [0, 1],
                        'count': 2,
                        'mean_y': 0.4,
                        'connections': []
                    },
                    {
                        'id': 'B1',
                        'location': [0.8],
                        'point_indices': [2, 3, 4],
                        'count': 3,
                        'mean_y': 0.6,
                        'connections': []
                    }
                ]
            }
        }
    )
    
    fig = visualize.draw_tdabm(corpus=corpus, show=False)
    
    assert fig is not None
    plt.close(fig)


def test_draw_graph_basic(visualize: QRVisualize):
    """Test basic graph visualization"""
    from src.crisp_t.model.corpus import Corpus
    from src.crisp_t.model.document import Document
    
    # Create test corpus with graph metadata
    doc1 = Document(
        id="doc1",
        name="Doc 1",
        text="Test document 1",
        metadata={"keywords": ["health", "research"]}
    )
    doc2 = Document(
        id="doc2",
        name="Doc 2",
        text="Test document 2",
        metadata={"keywords": ["health", "policy"]}
    )
    
    corpus = Corpus(
        id="test",
        name="Test",
        documents=[doc1, doc2],
        metadata={
            'graph': {
                'nodes': [
                    {'id': 'doc1', 'label': 'document', 'properties': {'name': 'Doc 1'}},
                    {'id': 'doc2', 'label': 'document', 'properties': {'name': 'Doc 2'}},
                    {'id': 'keyword:health', 'label': 'keyword', 'properties': {'name': 'health'}},
                    {'id': 'keyword:research', 'label': 'keyword', 'properties': {'name': 'research'}},
                    {'id': 'keyword:policy', 'label': 'keyword', 'properties': {'name': 'policy'}},
                ],
                'edges': [
                    {'source': 'doc1', 'target': 'keyword:health', 'label': 'HAS_KEYWORD', 'properties': {}},
                    {'source': 'doc1', 'target': 'keyword:research', 'label': 'HAS_KEYWORD', 'properties': {}},
                    {'source': 'doc2', 'target': 'keyword:health', 'label': 'HAS_KEYWORD', 'properties': {}},
                    {'source': 'doc2', 'target': 'keyword:policy', 'label': 'HAS_KEYWORD', 'properties': {}},
                ],
                'num_nodes': 5,
                'num_edges': 4,
                'num_documents': 2,
                'has_keywords': True,
                'has_clusters': False,
                'has_metadata': False
            }
        }
    )
    
    fig = visualize.draw_graph(corpus=corpus, show=False)
    
    assert fig is not None
    plt.close(fig)


def test_draw_graph_without_graph_metadata(visualize: QRVisualize):
    """Test graph visualization fails without graph metadata"""
    from src.crisp_t.model.corpus import Corpus
    
    corpus = Corpus(id="test", name="Test")
    
    with pytest.raises(ValueError, match="does not contain 'graph' data"):
        visualize.draw_graph(corpus=corpus, show=False)


def test_draw_graph_without_corpus(visualize: QRVisualize):
    """Test graph visualization fails without corpus"""
    
    with pytest.raises(ValueError, match="No corpus provided"):
        visualize.draw_graph(corpus=None, show=False)


def test_draw_graph_with_different_layouts(visualize: QRVisualize):
    """Test graph visualization with different layout algorithms"""
    from src.crisp_t.model.corpus import Corpus
    from src.crisp_t.model.document import Document
    
    # Create simple graph
    doc1 = Document(
        id="doc1",
        name="Doc 1",
        text="Test",
        metadata={"keywords": ["test"]}
    )
    
    corpus = Corpus(
        id="test",
        name="Test",
        documents=[doc1],
        metadata={
            'graph': {
                'nodes': [
                    {'id': 'doc1', 'label': 'document', 'properties': {'name': 'Doc 1'}},
                    {'id': 'keyword:test', 'label': 'keyword', 'properties': {'name': 'test'}},
                ],
                'edges': [
                    {'source': 'doc1', 'target': 'keyword:test', 'label': 'HAS_KEYWORD', 'properties': {}},
                ],
                'num_nodes': 2,
                'num_edges': 1,
                'num_documents': 1,
                'has_keywords': True,
                'has_clusters': False,
                'has_metadata': False
            }
        }
    )
    
    # Test different layouts
    for layout in ["spring", "circular", "kamada_kawai", "spectral"]:
        fig = visualize.draw_graph(corpus=corpus, show=False, layout=layout)
        assert fig is not None
        plt.close(fig)
