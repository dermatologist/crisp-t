import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from src.crisp_t.cli import main


def write_sample_corpus(tmp_path, folder_path_fixture):
    # Use packaged resources to build a corpus on disk
    # Create a minimal corpus.json and corpus_df.csv using the CLI read_source path
    # We'll invoke the CLI to read from folder and write to tmp_path
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--source",
            folder_path_fixture,
            "--out",
            str(tmp_path / "out.json"),
        ],
    )
    assert result.exit_code == 0
    assert (tmp_path / "corpus.json").exists()
    delete_corpus_df_in_temp_path(tmp_path)
    return tmp_path


def test_filters_key_value_with_source(tmp_path, folder_path_fixture):
    runner = CliRunner()
    # Expect filter by file_name based on resource file available
    result = runner.invoke(
        main,
        [
            "--source",
            folder_path_fixture,
            "--filters",
            "file_name=sample.txt",
        ],
    )
    assert result.exit_code == 0
    # Should print remaining documents after filtering
    delete_corpus_df_in_temp_path(tmp_path)
    assert "Applied filters" in result.output


def test_filters_invalid_format(tmp_path, folder_path_fixture):
    runner = CliRunner()
    result = runner.invoke(
        main, ["--source", folder_path_fixture, "--filters", "file_name:sample.txt"]
    )
    assert result.exit_code == 0
    assert "Filters are not supported when using --source" in result.output
    delete_corpus_df_in_temp_path(tmp_path)


def test_filters_with_inp_and_save(tmp_path, folder_path_fixture):
    # First, create a corpus by reading from resources and writing to tmp dir
    runner = CliRunner()
    result1 = runner.invoke(
        main, ["--source", folder_path_fixture, "--out", str(tmp_path)]
    )
    assert result1.exit_code == 0
    assert (tmp_path / "corpus.json").exists()

    # Now load it via --inp and apply a filter, and save again
    result2 = runner.invoke(
        main,
        [
            "--inp",
            str(tmp_path),
            "--filters",
            "file_name=sample.txt",
            "--out",
            str(tmp_path / "save2.json"),
        ],
    )
    assert result2.exit_code == 0
    assert (tmp_path / "corpus.json").exists()
    delete_corpus_df_in_temp_path(tmp_path)


def delete_corpus_df_in_temp_path(tmp_path):
    corpus_df_path = Path(tmp_path) / "corpus_df.csv"
    if corpus_df_path.exists():
        os.remove(corpus_df_path)