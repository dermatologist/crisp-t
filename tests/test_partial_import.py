"""Tests for partial import functionality."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.crisp_t.read_data import ReadData


@pytest.fixture
def temp_source_dir():
    """Create a temporary directory with multiple text, PDF, and CSV files for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create 5 text files
        for i in range(5):
            text_file = tmpdir_path / f"text_{i}.txt"
            text_file.write_text(f"This is text file number {i}.")
        
        # Create CSV file with 20 rows
        csv_data = {
            'id': list(range(1, 21)),
            'value': [f'value_{i}' for i in range(1, 21)],
            'number': list(range(100, 120))
        }
        df = pd.DataFrame(csv_data)
        csv_file = tmpdir_path / "data.csv"
        df.to_csv(csv_file, index=False)
        
        yield str(tmpdir_path)


def test_partial_text_file_import(temp_source_dir):
    """Test limiting the number of text files imported."""
    read_data = ReadData()
    
    # Import only 2 text files
    read_data.read_source(temp_source_dir, max_text_files=2)
    corpus = read_data.create_corpus()
    
    # Should only have 2 text documents (CSV not counted as document)
    assert len(corpus.documents) == 2
    assert corpus.df is not None
    assert len(corpus.df) == 20  # CSV should be fully imported


def test_partial_csv_row_import(temp_source_dir):
    """Test limiting the number of CSV rows imported."""
    read_data = ReadData()
    
    # Import only 5 CSV rows
    read_data.read_source(temp_source_dir, max_csv_rows=5)
    corpus = read_data.create_corpus()
    
    # Should have all 5 text files as documents
    assert len(corpus.documents) == 5
    # CSV should only have 5 rows
    assert corpus.df is not None
    assert len(corpus.df) == 5


def test_partial_import_both_limits(temp_source_dir):
    """Test limiting both text files and CSV rows."""
    read_data = ReadData()
    
    # Import only 3 text files and 10 CSV rows
    read_data.read_source(temp_source_dir, max_text_files=3, max_csv_rows=10)
    corpus = read_data.create_corpus()
    
    # Should have 3 text documents
    assert len(corpus.documents) == 3
    # CSV should have 10 rows
    assert corpus.df is not None
    assert len(corpus.df) == 10


def test_no_limits_full_import(temp_source_dir):
    """Test that without limits, all files are imported."""
    read_data = ReadData()
    
    # Import without limits
    read_data.read_source(temp_source_dir)
    corpus = read_data.create_corpus()
    
    # Should have all 5 text files
    assert len(corpus.documents) == 5
    # CSV should have all 20 rows
    assert corpus.df is not None
    assert len(corpus.df) == 20


def test_limit_greater_than_available(temp_source_dir):
    """Test that limits greater than available files work correctly."""
    read_data = ReadData()
    
    # Set limit higher than available files
    read_data.read_source(temp_source_dir, max_text_files=100, max_csv_rows=100)
    corpus = read_data.create_corpus()
    
    # Should import all available files (not raise error)
    assert len(corpus.documents) == 5
    assert corpus.df is not None
    assert len(corpus.df) == 20


def test_zero_limit(temp_source_dir):
    """Test that zero limit results in no files being imported."""
    read_data = ReadData()
    
    # Set limit to 0
    read_data.read_source(temp_source_dir, max_text_files=0, max_csv_rows=0)
    
    # create_corpus will raise an error when there are no documents
    with pytest.raises(ValueError, match="No documents found"):
        corpus = read_data.create_corpus()
    
    # Verify that no documents were added
    assert len(read_data._documents) == 0
    # CSV should have 0 rows
    assert read_data._df is not None
    assert len(read_data._df) == 0


@pytest.fixture
def temp_csv_text_columns_dir():
    """Create a temporary directory with CSV containing text columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create CSV with text columns (for read_csv_to_corpus)
        csv_data = {
            'id': list(range(1, 11)),
            'comment': [f'Comment text {i}' for i in range(1, 11)],
            'feedback': [f'Feedback text {i}' for i in range(1, 11)]
        }
        df = pd.DataFrame(csv_data)
        csv_file = tmpdir_path / "text_data.csv"
        df.to_csv(csv_file, index=False)
        
        yield str(tmpdir_path)


def test_partial_csv_to_corpus_import(temp_csv_text_columns_dir):
    """Test limiting rows when importing CSV as text documents."""
    read_data = ReadData()
    
    # Import CSV as text with row limit
    read_data.read_source(
        temp_csv_text_columns_dir,
        comma_separated_text_columns="comment,feedback",
        max_csv_rows=5
    )
    corpus = read_data.create_corpus()
    
    # Should only create documents from 5 CSV rows
    # Each row creates one document combining the text columns
    assert len(corpus.documents) == 5


def test_partial_csv_to_corpus_no_limit(temp_csv_text_columns_dir):
    """Test full import when no limit specified for CSV as text documents."""
    read_data = ReadData()
    
    # Import CSV as text without row limit
    read_data.read_source(
        temp_csv_text_columns_dir,
        comma_separated_text_columns="comment,feedback"
    )
    corpus = read_data.create_corpus()
    
    # Should create documents from all 10 CSV rows
    assert len(corpus.documents) == 10
