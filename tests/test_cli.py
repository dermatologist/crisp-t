import pytest
import tempfile
import pathlib
from click.testing import CliRunner
from src.crisp_t.cli import main
from src.crisp_t.model.corpus import Corpus
from src.crisp_t.model.document import Document
from src.crisp_t.read_data import ReadData
import pandas as pd


@pytest.fixture
def create_test_corpus():
    """Fixture to create test corpus with optional configurations."""
    def _create_corpus(with_df=False, with_metadata=False, with_metadata_cols=False):
        corpus = Corpus(id='test', name='Test', description='Test corpus')
        corpus.add_document(Document(id='1', text='Test', metadata={}))
        
        if with_df:
            df_data = {'id': [1, 2, 3], 'value': [10, 20, 30]}
            if with_metadata_cols:
                df_data['metadata_source'] = ['s1', 's2', 's3']
            corpus.df = pd.DataFrame(df_data)
        
        if with_metadata:
            corpus.metadata['pca'] = {'explained_variance': [0.4, 0.3, 0.2]}
        
        return corpus
    return _create_corpus


def test_cli_help():
    """Test that CLI help works."""
    runner = CliRunner()
    result = runner.invoke(main, ['--help'])
    assert result.exit_code == 0
    assert "CRISP-T: Cross Industry Standard Process for Triangulation" in result.output
    assert "--inp" in result.output
    assert "--csv" in result.output
    assert "--codedict" in result.output


def test_cli_no_input():
    """Test CLI behavior with no input."""
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code == 0
    assert "No input data provided" in result.output


def test_cli_csv_analysis():
    """Test CLI with CSV input."""
    runner = CliRunner()

    # Use the existing test CSV file
    csv_file = "tests/resources/food_coded.csv"

    result = runner.invoke(main, [
        '--csv', csv_file,
        '--unstructured', 'comfort_food,comfort_food_reasons',
        '--sentiment'
    ])

    assert result.exit_code == 0
    assert "CRISP-T" in result.output
    assert "--csv option has been deprecated" in result.output
    # assert "=== Sentiment Analysis ===" in result.output


@pytest.mark.skipif(True, reason="ML dependencies not available in test environment")
def test_cli_ml_functionality():
    """Test ML functionality (if available)."""
    runner = CliRunner()

    csv_file = "src/crisp_t/resources/vis/numeric.csv"

    result = runner.invoke(main, [
        '--csv', csv_file,
        '--titles', 'target_column',
        '--kmeans',
        '--num', '3'
    ])

    # This test would only pass if ML dependencies are installed
    assert "=== K-Means Clustering ===" in result.output or "ML dependencies" in result.output


def test_cli_print_documents():
    """Test --print documents option."""
    runner = CliRunner()
    result = runner.invoke(main, [
        '--inp', 'tests/resources',
        '--print', 'documents'
    ])
    
    assert result.exit_code == 0
    assert "=== Documents ===" in result.output
    assert "Total documents:" in result.output
    assert "Showing first 5 document(s)" in result.output


def test_cli_print_documents_with_count():
    """Test --print documents N option."""
    runner = CliRunner()
    result = runner.invoke(main, [
        '--inp', 'tests/resources',
        '--print', 'documents 3'
    ])
    
    assert result.exit_code == 0
    assert "=== Documents ===" in result.output
    assert "Showing first 3 document(s)" in result.output


def test_cli_print_documents_metadata():
    """Test --print documents metadata option."""
    runner = CliRunner()
    result = runner.invoke(main, [
        '--inp', 'tests/resources',
        '--print', 'documents metadata'
    ])
    
    assert result.exit_code == 0
    assert "=== Document Metadata ===" in result.output


def test_cli_print_metadata(create_test_corpus):
    """Test --print metadata option."""
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus = create_test_corpus()
        corpus.metadata['test_key'] = 'test_value'
        
        read_data = ReadData(corpus=corpus)
        read_data.write_corpus_to_json(tmpdir, corpus=corpus)
        
        result = runner.invoke(main, [
            '--inp', tmpdir,
            '--print', 'metadata'
        ])
        
        assert result.exit_code == 0
        assert "=== Corpus Metadata ===" in result.output


def test_cli_print_specific_metadata(create_test_corpus):
    """Test --print metadata KEY option."""
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus = create_test_corpus(with_metadata=True)
        
        read_data = ReadData(corpus=corpus)
        read_data.write_corpus_to_json(tmpdir, corpus=corpus)
        
        result = runner.invoke(main, [
            '--inp', tmpdir,
            '--print', 'metadata pca'
        ])
        
        assert result.exit_code == 0
        assert "=== Metadata: pca ===" in result.output
        assert "explained_variance" in result.output


def test_cli_print_dataframe(create_test_corpus):
    """Test --print dataframe option."""
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus = create_test_corpus(with_df=True)
        
        read_data = ReadData(corpus=corpus)
        read_data.write_corpus_to_json(tmpdir, corpus=corpus)
        
        result = runner.invoke(main, [
            '--inp', tmpdir,
            '--print', 'dataframe'
        ])
        
        assert result.exit_code == 0
        assert "=== DataFrame ===" in result.output
        assert "Shape:" in result.output


def test_cli_print_dataframe_metadata(create_test_corpus):
    """Test --print dataframe metadata option."""
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus = create_test_corpus(with_df=True, with_metadata_cols=True)
        
        read_data = ReadData(corpus=corpus)
        read_data.write_corpus_to_json(tmpdir, corpus=corpus)
        
        result = runner.invoke(main, [
            '--inp', tmpdir,
            '--print', 'dataframe metadata'
        ])
        
        assert result.exit_code == 0
        assert "=== DataFrame Metadata Columns ===" in result.output
        assert "metadata_source" in result.output


def test_cli_print_dataframe_stats(create_test_corpus):
    """Test --print dataframe stats option."""
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus = create_test_corpus(with_df=True)
        
        read_data = ReadData(corpus=corpus)
        read_data.write_corpus_to_json(tmpdir, corpus=corpus)
        
        result = runner.invoke(main, [
            '--inp', tmpdir,
            '--print', 'dataframe stats'
        ])
        
        assert result.exit_code == 0
        assert "=== DataFrame Statistics ===" in result.output
        assert "Distinct values per column:" in result.output


def test_cli_print_stats_deprecated(create_test_corpus):
    """Test --print stats option (deprecated)."""
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus = create_test_corpus(with_df=True)
        
        read_data = ReadData(corpus=corpus)
        read_data.write_corpus_to_json(tmpdir, corpus=corpus)
        
        result = runner.invoke(main, [
            '--inp', tmpdir,
            '--print', 'stats'
        ])
        
        assert result.exit_code == 0
        assert "deprecated" in result.output
        assert "=== DataFrame Statistics ===" in result.output


def test_cli_print_unquoted_syntax_documents():
    """Test --print with unquoted syntax for documents N."""
    runner = CliRunner()
    result = runner.invoke(main, [
        '--inp', 'tests/resources',
        '--print', 'documents',
        '--print', '3'
    ])
    
    assert result.exit_code == 0
    assert "=== Documents ===" in result.output
    assert "Showing first 3 document(s)" in result.output


def test_cli_print_unquoted_syntax_documents_metadata():
    """Test --print with unquoted syntax for documents metadata."""
    runner = CliRunner()
    result = runner.invoke(main, [
        '--inp', 'tests/resources',
        '--print', 'documents',
        '--print', 'metadata'
    ])
    
    assert result.exit_code == 0
    assert "=== Document Metadata ===" in result.output


def test_cli_print_unquoted_syntax_dataframe_stats(create_test_corpus):
    """Test --print with unquoted syntax for dataframe stats."""
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus = create_test_corpus(with_df=True)
        
        read_data = ReadData(corpus=corpus)
        read_data.write_corpus_to_json(tmpdir, corpus=corpus)
        
        result = runner.invoke(main, [
            '--inp', tmpdir,
            '--print', 'dataframe',
            '--print', 'stats'
        ])
        
        assert result.exit_code == 0
        assert "=== DataFrame Statistics ===" in result.output


def test_cli_print_unquoted_syntax_metadata_key(create_test_corpus):
    """Test --print with unquoted syntax for specific metadata key."""
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus = create_test_corpus(with_metadata=True)
        
        read_data = ReadData(corpus=corpus)
        read_data.write_corpus_to_json(tmpdir, corpus=corpus)
        
        result = runner.invoke(main, [
            '--inp', tmpdir,
            '--print', 'metadata',
            '--print', 'pca'
        ])
        
        assert result.exit_code == 0
        assert "=== Metadata: pca ===" in result.output
        assert "explained_variance" in result.output
