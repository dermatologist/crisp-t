import pandas as pd
import pytest

from src.crisp_t.model.corpus import Corpus
from src.crisp_t.model.document import Document


def make_document(doc_id, name, metadata=None):
    return Document(
        id=doc_id,
        name=name,
        description=None,
        score=0.0,
        text=f"Text for {name}",
        metadata=metadata or {},
    )


def make_corpus():
    docs = [make_document(f"doc{i}", f"Document {i}") for i in range(3)]
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    return Corpus(
        id="corpus1",
        name="Test Corpus",
        description="desc",
        score=None,
        documents=docs,
        df=df,
    )


def test_pretty_print(capsys):
    corpus = make_corpus()
    corpus.pretty_print()
    out = capsys.readouterr().out
    assert "corpus1" in out # fix color codes are added to output
    assert "Document 0" in out
    assert "DataFrame" in out
    assert "Corpus Metadata" in out


def test_get_all_df_column_names():
    corpus = make_corpus()
    assert corpus.get_all_df_column_names() == ["A", "B"]


def test_get_row_count():
    corpus = make_corpus()
    assert corpus.get_row_count() == 3


def test_get_row_by_index():
    corpus = make_corpus()
    row = corpus.get_row_by_index(1)
    assert row is not None
    assert row["A"] == 2
    assert row["B"] == 5
    assert corpus.get_row_by_index(10) is None


def test_get_all_document_ids():
    corpus = make_corpus()
    assert corpus.get_all_document_ids() == ["doc0", "doc1", "doc2"]


def test_get_document_by_id():
    corpus = make_corpus()
    doc = corpus.get_document_by_id("doc1")
    assert doc is not None
    assert doc.name == "Document 1"
    assert corpus.get_document_by_id("notfound") is None


def test_add_and_remove_document():
    corpus = make_corpus()
    new_doc = make_document("docX", "New Doc")
    corpus.add_document(new_doc)
    assert "docX" in corpus.get_all_document_ids()
    corpus.remove_document_by_id("docX")
    assert "docX" not in corpus.get_all_document_ids()


def test_update_metadata():
    corpus = make_corpus()
    corpus.update_metadata("foo", 123)
    assert corpus.metadata["foo"] == 123


def test_relationships():
    corpus = make_corpus()
    corpus.add_relationship("text:foo", "num:bar", "correlates")
    rels = corpus.get_relationships()
    assert len(rels) == 1
    assert rels[0]["relation"] == "correlates"
    corpus.clear_relationships()
    assert corpus.get_relationships() == []


def test_visualization_and_metadata():
    corpus = make_corpus()
    corpus.visualization["plot"] = "data"
    assert "plot" in corpus.visualization
    corpus.metadata["extra"] = "value"
    assert corpus.metadata["extra"] == "value"
