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
