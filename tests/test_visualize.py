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
