import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from src.crisp_t.vizcli import main as viz_main


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def text_source(tmp_dir):
    # Prepare a small text-only source folder
    d = Path(tmp_dir) / "src"
    d.mkdir(parents=True, exist_ok=True)
    (d / "a.txt").write_text("hello world this is a document")
    (d / "b.txt").write_text("another short document with words words")
    return str(d)


def test_vizcli_freq_and_top_terms(text_source, tmp_dir):
    runner = CliRunner()
    out_dir = Path(tmp_dir) / "out"
    result = runner.invoke(
        viz_main,
        [
            "--source",
            text_source,
            "--out",
            str(out_dir),
            "--freq",
            "--top-terms",
            "--top-n",
            "5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "word_frequency.png").exists()
    assert (out_dir / "top_terms.png").exists()


def test_vizcli_topics_and_wordcloud(text_source, tmp_dir):
    runner = CliRunner()
    out_dir = Path(tmp_dir) / "out2"
    result = runner.invoke(
        viz_main,
        [
            "--source",
            text_source,
            "--out",
            str(out_dir),
            "--topics-num",
            "2",
            "--by-topic",
            "--wordcloud",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "by_topic.png").exists()
    assert (out_dir / "wordcloud.png").exists()


def test_vizcli_ldavis(text_source, tmp_dir):
    """Test that --ldavis flag creates HTML visualization"""
    runner = CliRunner()
    out_dir = Path(tmp_dir) / "out_ldavis"
    result = runner.invoke(
        viz_main,
        [
            "--source",
            text_source,
            "--out",
            str(out_dir),
            "--topics-num",
            "2",
            "--ldavis",
        ],
    )
    assert result.exit_code == 0, result.output
    # Check if HTML file was created (or at least the command completed)
    # The file might not be created if pyLDAvis is not installed, but command should not fail
    assert "lda_visualization.html" in result.output or "Warning" in result.output


def test_vizcli_corr_heatmap_with_sources(tmp_dir):
    # Create minimal CSV-based source: a folder with corpus_df created via ReadData.write
    # For simplicity here, emulate via corpus with df is complex; we instead create a temp folder
    # with numeric CSV and a dummy text file; read_source in project picks up CSV into corpus.df
    src = Path(tmp_dir) / "src3"
    src.mkdir(parents=True, exist_ok=True)
    # numeric CSV
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
    df.to_csv(src / "data.csv", index=False)
    (src / "x.txt").write_text("text")

    runner = CliRunner()
    out_dir = Path(tmp_dir) / "out3"
    result = runner.invoke(
        viz_main,
        [
            "--sources",
            str(src),
            "--out",
            str(out_dir),
            "--corr-heatmap",
            "--corr-columns",
            "a,b",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "corr_heatmap.png").exists()


def test_vizcli_tdabm_without_metadata(tmp_dir):
    """Test TDABM visualization fails gracefully without metadata"""
    from src.crisp_t.model.corpus import Corpus
    from src.crisp_t.read_data import ReadData
    
    # Create a corpus without TDABM metadata
    corpus_dir = Path(tmp_dir) / "corpus"
    corpus_dir.mkdir()
    
    corpus = Corpus(
        id="test",
        name="Test Corpus",
        df=pd.DataFrame({
            'x1': [1, 2, 3],
            'x2': [4, 5, 6],
            'y': [7, 8, 9]
        })
    )
    
    rd = ReadData(corpus=corpus)
    rd.write_corpus_to_json(str(corpus_dir), corpus=corpus)
    
    runner = CliRunner()
    out_dir = Path(tmp_dir) / "out_tdabm"
    result = runner.invoke(
        viz_main,
        [
            "--inp",
            str(corpus_dir),
            "--out",
            str(out_dir),
            "--tdabm",
        ],
    )
    
    # Should not crash but should show warning
    assert result.exit_code == 0
    assert "No TDABM data found" in result.output or "Warning" in result.output


def test_vizcli_tdabm_with_metadata(tmp_dir):
    """Test TDABM visualization with proper metadata"""
    from src.crisp_t.model.corpus import Corpus
    from src.crisp_t.read_data import ReadData
    
    # Create a corpus with TDABM metadata
    corpus_dir = Path(tmp_dir) / "corpus_tdabm"
    corpus_dir.mkdir()
    
    corpus = Corpus(
        id="test",
        name="Test Corpus",
        df=pd.DataFrame({
            'x1': [1, 2, 3],
            'x2': [4, 5, 6],
            'y': [7, 8, 9]
        }),
        metadata={
            'tdabm': {
                'y_variable': 'y',
                'x_variables': ['x1', 'x2'],
                'radius': 0.3,
                'num_landmarks': 2,
                'num_points': 3,
                'landmarks': [
                    {
                        'id': 'B0',
                        'location': [0.2, 0.3],
                        'point_indices': [0, 1],
                        'count': 2,
                        'mean_y': 0.5,
                        'connections': ['B1']
                    },
                    {
                        'id': 'B1',
                        'location': [0.8, 0.7],
                        'point_indices': [1, 2],
                        'count': 2,
                        'mean_y': 0.8,
                        'connections': ['B0']
                    }
                ]
            }
        }
    )
    
    rd = ReadData(corpus=corpus)
    rd.write_corpus_to_json(str(corpus_dir), corpus=corpus)
    
    runner = CliRunner()
    out_dir = Path(tmp_dir) / "out_tdabm_viz"
    result = runner.invoke(
        viz_main,
        [
            "--inp",
            str(corpus_dir),
            "--out",
            str(out_dir),
            "--tdabm",
        ],
    )
    
    assert result.exit_code == 0, result.output
    assert (out_dir / "tdabm.png").exists()
