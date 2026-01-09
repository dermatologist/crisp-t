"""
Test CLI read_source functionality.
"""

import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from src.crisp_t.cli import main


@pytest.fixture
def test_directory():
    """Create a temporary directory with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test text files
        test_file1 = Path(tmpdir) / "test1.txt"
        test_file2 = Path(tmpdir) / "test2.txt"

        test_file1.write_text("This is test document number one.")
        test_file2.write_text("This is test document number two.")

        yield tmpdir


@pytest.fixture
def output_directory():
    """Create a temporary directory for output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_read_source_from_directory(test_directory):
    """Test reading source data from a directory."""
    runner = CliRunner()
    result = runner.invoke(main, ["--source", test_directory])

    assert result.exit_code == 0
    assert "Reading data from source:" in result.output
    assert "Successfully loaded" in result.output
    assert "document(s)" in result.output


def test_read_source_with_output(test_directory, output_directory):
    """Test reading source and saving corpus."""
    runner = CliRunner()
    output_path = Path(output_directory) / "output" / "corpus.json"

    result = runner.invoke(main, ["--source", test_directory, "--out", str(output_path)])

    assert result.exit_code == 0
    assert "Successfully loaded" in result.output
    assert "CRISP-T" in result.output

    # Check if corpus.json was created
    corpus_file = Path(output_directory) / "output" / "corpus.json"
    assert corpus_file.exists()


def test_read_source_with_ignore_words(test_directory):
    """Test reading source with ignore words."""
    runner = CliRunner()
    result = runner.invoke(main, ["--source", test_directory, "--ignore", "test,number"])

    assert result.exit_code == 0
    assert "Successfully loaded" in result.output


def test_read_source_nonexistent_directory():
    """Test error handling for nonexistent directory."""
    runner = CliRunner()
    result = runner.invoke(main, ["--source", "/nonexistent/directory"])

    # Should not crash but may have error message
    # The exact behavior depends on read_source implementation
    assert result.exit_code == 0  # CLI should handle gracefully


def test_read_source_verbose(test_directory):
    """Test reading source with verbose output."""
    runner = CliRunner()
    result = runner.invoke(main, ["--source", test_directory, "--verbose"])

    assert result.exit_code == 0
    assert "Verbose mode enabled" in result.output
    assert "Successfully loaded" in result.output


def test_read_source_url_format():
    """Test that URL format is accepted (may not work without network)."""
    runner = CliRunner()
    # Using example.com which should be accessible
    result = runner.invoke(main, ["--source", "https://example.com"])

    # Should not crash, even if it fails to read
    assert result.exit_code == 0


def test_import_flag_from_directory(test_directory):
    """Test reading source data using --import flag."""
    runner = CliRunner()
    result = runner.invoke(main, ["--import", test_directory])

    assert result.exit_code == 0
    assert "Reading data from source:" in result.output
    assert "Successfully loaded" in result.output
    assert "document(s)" in result.output


def test_import_flag_with_output(test_directory, output_directory):
    """Test reading source using --import flag and saving corpus."""
    runner = CliRunner()
    output_path = Path(output_directory) / "output" / "corpus.json"

    result = runner.invoke(main, ["--import", test_directory, "--out", str(output_path)])

    assert result.exit_code == 0
    assert "Successfully loaded" in result.output
    assert "CRISP-T" in result.output

    # Check if corpus.json was created
    corpus_file = Path(output_directory) / "output" / "corpus.json"
    assert corpus_file.exists()


def test_import_flag_with_ignore_words(test_directory):
    """Test reading source using --import flag with ignore words."""
    runner = CliRunner()
    result = runner.invoke(main, ["--import", test_directory, "--ignore", "test,number"])

    assert result.exit_code == 0
    assert "Successfully loaded" in result.output


def test_help_contains_data_preparation_steps():
    """Test that help output includes data preparation steps."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "TIPS" in result.output
    assert "GETTING STARTED" in result.output
    assert "Create a source directory" in result.output
    assert "Add your data files to this directory:" in result.output
    assert "Import your data to create a corpus" in result.output


def test_help_shows_import_flag():
    """Test that help output shows --import flag."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "--source, --import" in result.output or "--import, --source" in result.output
