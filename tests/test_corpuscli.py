import os
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from src.crisp_t.corpuscli import main as corpus_main


def run_cli(args, tmp_path=None):
    runner = CliRunner()
    if tmp_path:
        with runner.isolated_filesystem():
            # If any file args, rewrite to tmp_path
            args = [
                (
                    str(tmp_path / a)
                    if a
                    and (a.endswith(".json") or a.endswith(".csv") or os.path.isdir(a))
                    else a
                )
                for a in args
            ]
            return runner.invoke(corpus_main, args)
    return runner.invoke(corpus_main, args)


def test_save_and_load_corpus(tmp_path):
    # Save corpus
    out_dir = tmp_path / "corpus_save"
    out_dir.mkdir()
    out_path = out_dir / "corpus.json"
    result = run_cli(
        [
            "--id",
            "corp9",
            "--name",
            "SaveTest",
            "--doc",
            "d1|Doc 1|Text",
            "--out",
            str(out_dir),
        ],
        tmp_path=tmp_path,
    )
    assert result.exit_code == 0, result.output
    assert "✓ Corpus saved to" in result.output
    # File exists
    assert (out_dir / "corpus.json").exists()

    # Load corpus
    result2 = run_cli(["--inp", str(out_dir), "--print"], tmp_path=tmp_path)
    assert result2.exit_code == 0, result2.output
    assert "✓ Loaded corpus from" in result2.output
    assert "Corpus ID: corp9" in result2.output
    assert "Name: SaveTest" in result2.output
    assert "ID: d1" in result2.output


def test_create_and_print_corpus(capsys):
    result = run_cli(
        [
            "--id",
            "corp1",
            "--name",
            "My Corpus",
            "--description",
            "A test corpus",
            "--print",
        ]
    )
    assert result.exit_code == 0, result.output
    # Printed banner and fields
    assert "CRISP-T: Corpus CLI" in result.output
    assert "Corpus ID: corp1" in result.output
    assert "Name: My Corpus" in result.output
    assert "Description: A test corpus" in result.output


def test_add_documents_and_list_ids():
    result = run_cli(
        [
            "--id",
            "corp2",
            "--doc",
            "d1|Doc 1|Hello world",
            "--doc",
            "d2|Doc 2|Another text",
            "--print",
        ]
    )
    assert result.exit_code == 0, result.output
    assert "✓ Added 2 document(s)" in result.output
    # pretty_print lists each doc id
    assert "ID: d1" in result.output
    assert "ID: d2" in result.output


def test_remove_document():
    result = run_cli(
        [
            "--id",
            "corp3",
            "--doc",
            "d1|Doc 1|Hello",
            "--doc",
            "d2|Doc 2|World",
            "--remove-doc",
            "d1",
            "--print",
        ]
    )
    assert result.exit_code == 0, result.output
    assert "✓ Removed 1 document(s)" in result.output
    assert "ID: d1" not in result.output
    assert "ID: d2" in result.output


def test_update_metadata_and_print():
    result = run_cli(
        [
            "--id",
            "corp4",
            "--meta",
            "owner=alice",
            "--meta",
            "project=test",
            "--print",
        ]
    )
    assert result.exit_code == 0, result.output
    assert "✓ Updated metadata entries: 2" in result.output
    # pretty_print shows metadata in lines formatted as ' - key\n: value'
    assert " - owner" in result.output
    assert ": alice" in result.output
    assert " - project" in result.output
    assert ": test" in result.output


def test_add_and_clear_relationships():
    result = run_cli(
        [
            "--id",
            "corp5",
            "--add-rel",
            "text:foo|num:bar|correlates",
            "--add-rel",
            "text:baz|text:qux|references",
            "--print",
            "--clear-rel",
        ]
    )
    assert result.exit_code == 0, result.output
    assert "✓ Added 2 relationship(s)" in result.output
    # There is a 'Visualization:' section printed as keys view; ensure CLI ran
    assert "Visualization:" in result.output
    # Clear relationships confirmation
    assert "✓ Cleared relationships" in result.output


def test_invalid_meta_format():
    result = run_cli(
        [
            "--id",
            "corp6",
            "--meta",
            "badformat",
        ]
    )
    assert result.exit_code != 0
    assert "Invalid metadata" in result.output


def test_invalid_relationship_format():
    result = run_cli(
        [
            "--id",
            "corp7",
            "--add-rel",
            "onlytwo|parts",
        ]
    )
    assert result.exit_code != 0
    assert "Invalid relationship" in result.output


def test_invalid_doc_format():
    result = run_cli(
        [
            "--id",
            "corp8",
            "--doc",
            "badformat",
        ]
    )
    assert result.exit_code != 0
    assert "Invalid --doc value" in result.output
