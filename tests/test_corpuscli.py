def test_relationships_for_keyword(tmp_path):
    out_dir = tmp_path / "corpus_rels_kw"
    out_dir.mkdir()
    from src.crisp_t.model.corpus import Corpus
    from src.crisp_t.model.document import Document

    corpus = Corpus(
        id="relidkw",
        name="relnamekw",
        description="reldesckw",
        score=None,
        documents=[
            Document(
                id="d1", name="n1", description=None, score=0.0, text="t1", metadata={}
            )
        ],
        df=None,
    )
    corpus.add_relationship("text:foo", "num:bar", "correlates")
    corpus.add_relationship("text:baz", "text:qux", "references")
    corpus.add_relationship("text:foo", "text:qux", "contradicts")
    from src.crisp_t.read_data import ReadData

    rd = ReadData(corpus=corpus)
    rd.write_corpus_to_json(out_dir, corpus=corpus)
    # Keyword 'foo' should match two relationships
    result = run_cli(
        ["--inp", str(out_dir), "--relationships-for-keyword", "foo"], tmp_path=tmp_path
    )
    assert result.exit_code == 0, result.output
    assert "Relationships for keyword 'foo':" in result.output
    # Should contain two relationships with 'foo'
    assert result.output.count("foo") >= 2


def test_df_column_names_and_row_count(tmp_path):
    # Save corpus with DataFrame
    import pandas as pd

    from src.crisp_t.model.corpus import Corpus
    from src.crisp_t.model.document import Document

    out_dir = tmp_path / "corpus_df"
    out_dir.mkdir()
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    corpus = Corpus(
        id="dfid",
        name="dfname",
        description="dfdesc",
        score=None,
        documents=[
            Document(
                id="d1", name="n1", description=None, score=0.0, text="t1", metadata={}
            )
        ],
        df=df,
    )
    from src.crisp_t.read_data import ReadData

    rd = ReadData(corpus=corpus)
    rd.write_corpus_to_json(out_dir, corpus=corpus)
    # Load and test CLI
    result = run_cli(
        ["--inp", str(out_dir), "--df-cols", "--df-row-count"], tmp_path=tmp_path
    )
    assert result.exit_code == 0, result.output
    assert "DataFrame columns: ['A', 'B']" in result.output
    assert "DataFrame row count: 2" in result.output


def test_df_row_by_index(tmp_path):
    import pandas as pd

    from src.crisp_t.model.corpus import Corpus
    from src.crisp_t.model.document import Document

    out_dir = tmp_path / "corpus_dfrow"
    out_dir.mkdir()
    df = pd.DataFrame({"A": [10, 20], "B": [30, 40]})
    corpus = Corpus(
        id="dfid2",
        name="dfname2",
        description="dfdesc2",
        score=None,
        documents=[
            Document(
                id="d2", name="n2", description=None, score=0.0, text="t2", metadata={}
            )
        ],
        df=df,
    )
    from src.crisp_t.read_data import ReadData

    rd = ReadData(corpus=corpus)
    rd.write_corpus_to_json(out_dir, corpus=corpus)
    # Valid index - test checks for key content (emoji format may vary between terminals)
    result = run_cli(["--inp", str(out_dir), "--df-row", "1"], tmp_path=tmp_path)
    assert result.exit_code == 0, result.output
    # Check for the key content, emoji may vary
    assert "DataFrame row 1:" in result.output
    assert "{'A': 20, 'B': 40}" in result.output
    # Invalid index - use partial text matching for robustness
    result2 = run_cli(["--inp", str(out_dir), "--df-row", "5"], tmp_path=tmp_path)
    assert result2.exit_code == 0, result2.output
    assert "No row" in result2.output and "index 5" in result2.output


def test_doc_ids_and_get_document(tmp_path):
    out_dir = tmp_path / "corpus_docs"
    out_dir.mkdir()
    from src.crisp_t.model.corpus import Corpus
    from src.crisp_t.model.document import Document

    docs = [
        Document(
            id="d1", name="n1", description=None, score=0.0, text="t1", metadata={}
        ),
        Document(
            id="d2", name="n2", description=None, score=0.0, text="t2", metadata={}
        ),
    ]
    corpus = Corpus(
        id="docid",
        name="docname",
        description="docdesc",
        score=None,
        documents=docs,
        df=None,
    )
    from src.crisp_t.read_data import ReadData

    rd = ReadData(corpus=corpus)
    rd.write_corpus_to_json(out_dir, corpus=corpus)
    # List IDs
    result = run_cli(["--inp", str(out_dir), "--doc-ids"], tmp_path=tmp_path)
    assert result.exit_code == 0, result.output
    assert "Document IDs: ['d1', 'd2']" in result.output
    # Get document by ID
    result2 = run_cli(["--inp", str(out_dir), "--doc-id", "d2"], tmp_path=tmp_path)
    assert result2.exit_code == 0, result2.output
    assert "Document d2:" in result2.output
    # Nonexistent ID
    result3 = run_cli(["--inp", str(out_dir), "--doc-id", "dX"], tmp_path=tmp_path)
    assert result3.exit_code == 0, result3.output
    assert "No document found with ID dX" in result3.output


def test_print_relationships(tmp_path):
    out_dir = tmp_path / "corpus_rels"
    out_dir.mkdir()
    from src.crisp_t.model.corpus import Corpus
    from src.crisp_t.model.document import Document

    corpus = Corpus(
        id="relid",
        name="relname",
        description="reldesc",
        score=None,
        documents=[
            Document(
                id="d1", name="n1", description=None, score=0.0, text="t1", metadata={}
            )
        ],
        df=None,
    )
    corpus.add_relationship("text:foo", "num:bar", "correlates")
    corpus.add_relationship("text:baz", "text:qux", "references")
    from src.crisp_t.read_data import ReadData

    rd = ReadData(corpus=corpus)
    rd.write_corpus_to_json(out_dir, corpus=corpus)
    result = run_cli(["--inp", str(out_dir), "--relationships"], tmp_path=tmp_path)
    assert result.exit_code == 0, result.output
    assert (
        "Relationships: [{'first': 'text:foo', 'second': 'num:bar', 'relation': 'correlates'}, {'first': 'text:baz', 'second': 'text:qux', 'relation': 'references'}]"
        in result.output
    )


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
    assert "Loading corpus from" in result2.output
    assert "ID:\x1b[0m d1" in result2.output # With color codes


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
    # Printed banner and fields - now without colon
    assert "CRISP-T Corpus CLI" in result.output


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
    assert "ID:\x1b[0m d1" in result.output
    assert "'id': 'd2'" in result.output


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
    assert "ID:\x1b[0m d2" in result.output


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
    # Changed text: "entries" to "entry/entries" depending on count
    assert "✓ Updated 2 metadata entr" in result.output  # matches both "entry" and "entries"
    # pretty_print shows metadata in lines formatted as ' - key\n: value'
    assert "owner" in result.output
    assert "alice" in result.output
    assert "project" in result.output
    assert "test" in result.output


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
    # Header changed to use box drawing characters
    assert "CORPUS DETAILS" in result.output
    # Clear relationships confirmation - text changed
    assert "✓ Cleared all relationships" in result.output


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
