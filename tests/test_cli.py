import pytest
import tempfile
import pathlib
from click.testing import CliRunner
from src.crisp_t.cli import main


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


def test_cli_text_file_analysis():
    """Test CLI with text file input."""
    runner = CliRunner()
    
    # Create a temporary text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document for analysis. It contains some text for coding and sentiment analysis.")
        temp_file = f.name
    
    try:
        result = runner.invoke(main, ['--inp', temp_file, '--codedict'])
        assert result.exit_code == 0
        assert "Loaded corpus with 1 documents" in result.output
        assert "=== Generating Coding Dictionary ===" in result.output
    finally:
        pathlib.Path(temp_file).unlink()


def test_cli_csv_analysis():
    """Test CLI with CSV input."""
    runner = CliRunner()
    
    # Use the existing test CSV file
    csv_file = "src/crisp_t/resources/food_coded.csv"
    
    result = runner.invoke(main, [
        '--csv', csv_file,
        '--titles', 'comfort_food,comfort_food_reasons',
        '--sentiment'
    ])
    
    assert result.exit_code == 0
    assert "Loaded CSV with shape" in result.output
    assert "Created corpus from CSV with" in result.output
    assert "=== Sentiment Analysis ===" in result.output


def test_cli_output_functionality():
    """Test CLI output saving functionality."""
    runner = CliRunner()
    
    # Create a temporary text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test document for output functionality.")
        temp_file = f.name
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = pathlib.Path(temp_dir) / "test_output"
        
        try:
            result = runner.invoke(main, [
                '--inp', temp_file,
                '--codedict',
                '--out', str(output_path)
            ])
            
            assert result.exit_code == 0
            assert "Results saved to:" in result.output
            
            # Check that output file was created
            expected_file = pathlib.Path(f"{output_path}_coding_dictionary.json")
            assert expected_file.exists()
            
        finally:
            pathlib.Path(temp_file).unlink()


def test_cli_verbose_mode():
    """Test CLI verbose mode."""
    runner = CliRunner()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test document for verbose mode.")
        temp_file = f.name
    
    try:
        result = runner.invoke(main, ['--inp', temp_file, '--codedict', '--verbose'])
        assert result.exit_code == 0
        assert "Verbose mode enabled" in result.output
    finally:
        pathlib.Path(temp_file).unlink()


def test_cli_nlp_comprehensive():
    """Test comprehensive NLP analysis."""
    runner = CliRunner()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a comprehensive test document. It should generate topics, categories, and summaries. The sentiment should be analyzed as well.")
        temp_file = f.name
    
    try:
        result = runner.invoke(main, ['--inp', temp_file, '--nlp', '--num', '2'])
        assert result.exit_code == 0
        assert "=== Generating Coding Dictionary ===" in result.output
        assert "=== Topic Modeling ===" in result.output
        assert "=== Document-Topic Assignments ===" in result.output
        assert "=== Category Analysis ===" in result.output
        assert "=== Text Summarization ===" in result.output
        assert "=== Sentiment Analysis ===" in result.output
    finally:
        pathlib.Path(temp_file).unlink()


def test_cli_invalid_input_file():
    """Test CLI behavior with non-existent input file."""
    runner = CliRunner()
    result = runner.invoke(main, ['--inp', 'nonexistent_file.txt'])
    assert result.exit_code == 0
    assert "Input file not found" in result.output


def test_cli_unsupported_file_format():
    """Test CLI behavior with unsupported file format."""
    runner = CliRunner()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
        f.write("Test content")
        temp_file = f.name
    
    try:
        result = runner.invoke(main, ['--inp', temp_file])
        assert result.exit_code == 0
        assert "Unsupported input file format" in result.output
    finally:
        pathlib.Path(temp_file).unlink()


@pytest.mark.skipif(True, reason="ML dependencies not available in test environment")
def test_cli_ml_functionality():
    """Test ML functionality (if available)."""
    runner = CliRunner()
    
    csv_file = "src/crisp_t/resources/numeric.csv"
    
    result = runner.invoke(main, [
        '--csv', csv_file,
        '--titles', 'target_column',
        '--kmeans',
        '--num', '3'
    ])
    
    # This test would only pass if ML dependencies are installed
    assert "=== K-Means Clustering ===" in result.output or "ML dependencies" in result.output