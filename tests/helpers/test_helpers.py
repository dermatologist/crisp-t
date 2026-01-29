import types

import pandas as pd
import pytest

from src.crisp_t.helpers import analyzer, initializer
from src.crisp_t.model import Document


class DummyCorpus:
    def __init__(self):
        self.documents = [
            Document(id="doc1", text="test document 1", metadata={}),
            Document(id="doc2", text="test document 2", metadata={}),
        ]
        self.df = None  # type: ignore


class DummyCsv:
    def __init__(self, corpus=None):
        self.corpus = corpus
        self.comma_separated_text_columns = ""
        self.comma_separated_ignore_columns = ""
        self.df = None

    def get_shape(self):
        if self.df is not None:
            return self.df.shape
        return (2, 2)

    def filter_rows_by_column_value(self, key, value):
        return None


class DummyText:
    def __init__(self, corpus=None):
        self.corpus = corpus

    def filter_documents(self, key, value):
        return None

    def document_count(self):
        if self.corpus and hasattr(self.corpus, "documents"):
            return len(self.corpus.documents)
        return 2


# Patch imports for analyzer
analyzer.Text = DummyText
analyzer.Csv = DummyCsv


# initializer.py tests
def test_initialize_corpus_source(monkeypatch):

    class DummyReadData:
        def read_source(
            self,
            source,
            comma_separated_text_columns=None,
            comma_separated_ignore_words=None,
            **kwargs,
        ):
            pass

        def create_corpus(self, name, description):
            class Corpus:
                documents = ["doc1", "doc2"]

            return Corpus()

    monkeypatch.setattr(initializer, "ReadData", DummyReadData)
    corpus = initializer.initialize_corpus(source="dummy_source")
    assert corpus.documents == ["doc1", "doc2"]


def test_initialize_corpus_inp(monkeypatch):
    class DummyReadData:
        def read_corpus_from_json(self, inp, comma_separated_ignore_words=None):
            class Corpus:
                documents = ["doc1", "doc2"]

            return Corpus()

    monkeypatch.setattr(initializer, "ReadData", DummyReadData)
    corpus = initializer.initialize_corpus(inp="dummy_inp")
    assert corpus.documents == ["doc1", "doc2"]


# analyzer.py tests
def test_get_text_analyzer_filters():
    corpus = DummyCorpus()
    filters = ["key=value", "foo:bar"]
    analyzer.Text = DummyText
    ta = analyzer.get_text_analyzer(corpus, filters=filters)
    assert isinstance(ta, DummyText)
    assert ta.corpus == corpus


def test_get_csv_analyzer(monkeypatch):
    corpus = DummyCorpus()
    corpus.df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    analyzer.Csv = DummyCsv
    csv_analyzer = analyzer.get_csv_analyzer(
        corpus, "text_col", "ignore_col", filters=["a=b"]
    )
    assert isinstance(csv_analyzer, DummyCsv)
    assert csv_analyzer.comma_separated_text_columns == "text_col"
    assert csv_analyzer.comma_separated_ignore_columns == "ignore_col"


def test_get_analyzers_basic():
    """Test get_analyzers returns both analyzers."""
    corpus = DummyCorpus()
    corpus.df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    analyzer.Text = DummyText
    analyzer.Csv = DummyCsv

    text_analyzer, csv_analyzer = analyzer.get_analyzers(corpus)

    assert isinstance(text_analyzer, DummyText)
    assert isinstance(csv_analyzer, DummyCsv)
    assert text_analyzer.corpus == corpus
    assert csv_analyzer.corpus == corpus


def test_get_analyzers_with_regular_filters():
    """Test get_analyzers with regular key=value filters."""
    corpus = DummyCorpus()
    corpus.df = pd.DataFrame(
        {"keywords": ["mask", "vaccine", "mask"], "value": [1, 2, 3]}
    )
    analyzer.Text = DummyText
    analyzer.Csv = DummyCsv

    filters = ["keywords=mask"]
    text_analyzer, csv_analyzer = analyzer.get_analyzers(corpus, filters=filters)

    assert isinstance(text_analyzer, DummyText)
    assert isinstance(csv_analyzer, DummyCsv)


def test_get_analyzers_with_embedding_filter():
    """Test get_analyzers with =embedding filter (legacy)."""
    corpus = DummyCorpus()
    # Add embedding_links to documents
    corpus.documents[0].metadata["embedding_links"] = [
        {"df_index": 0, "similarity": 0.9},
        {"df_index": 2, "similarity": 0.8},
    ]
    corpus.df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    analyzer.Text = DummyText
    analyzer.Csv = DummyCsv

    filters = ["=embedding"]
    text_analyzer, csv_analyzer = analyzer.get_analyzers(corpus, filters=filters)

    assert isinstance(text_analyzer, DummyText)
    assert isinstance(csv_analyzer, DummyCsv)
    # CSV should be filtered to only rows 0 and 2
    assert len(csv_analyzer.df) == 2
    assert 0 in csv_analyzer.df.index
    assert 2 in csv_analyzer.df.index


def test_get_analyzers_with_embedding_text_filter():
    """Test get_analyzers with embedding:text filter (explicit text→df direction)."""
    corpus = DummyCorpus()
    # Add embedding_links to documents
    corpus.documents[0].metadata["embedding_links"] = [
        {"df_index": 0, "similarity": 0.9},
        {"df_index": 2, "similarity": 0.8},
    ]
    corpus.df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    analyzer.Text = DummyText
    analyzer.Csv = DummyCsv

    filters = ["embedding:text"]
    text_analyzer, csv_analyzer = analyzer.get_analyzers(corpus, filters=filters)

    assert isinstance(text_analyzer, DummyText)
    assert isinstance(csv_analyzer, DummyCsv)
    # CSV should be filtered to only rows 0 and 2
    assert len(csv_analyzer.df) == 2
    assert 0 in csv_analyzer.df.index
    assert 2 in csv_analyzer.df.index


def test_get_analyzers_with_embedding_df_filter():
    """Test get_analyzers with embedding:df filter (df→text direction)."""
    corpus = DummyCorpus()
    corpus.documents[0].id = "doc1"
    corpus.documents[1].id = "doc2"
    # Add embedding_links to documents (reverse: which docs link to which rows)
    corpus.documents[0].metadata["embedding_links"] = [
        {"df_index": 0, "similarity": 0.9}
    ]
    corpus.documents[1].metadata["embedding_links"] = [
        {"df_index": 1, "similarity": 0.8}
    ]
    corpus.df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]}, index=[0, 1, 2])
    analyzer.Text = DummyText
    analyzer.Csv = DummyCsv

    filters = ["embedding:df"]
    text_analyzer, csv_analyzer = analyzer.get_analyzers(corpus, filters=filters)

    assert isinstance(text_analyzer, DummyText)
    assert isinstance(csv_analyzer, DummyCsv)
    # Documents should be filtered to only those linked from df rows 0 and 1
    assert len(text_analyzer.corpus.documents) == 2
    assert text_analyzer.corpus.documents[0].id == "doc1"
    assert text_analyzer.corpus.documents[1].id == "doc2"


def test_get_analyzers_with_temporal_filter():
    """Test get_analyzers with :temporal filter (legacy)."""
    corpus = DummyCorpus()
    # Add temporal_links to documents
    corpus.documents[1].metadata["temporal_links"] = [
        {"df_index": 1, "time_gap_seconds": 10}
    ]
    corpus.df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    analyzer.Text = DummyText
    analyzer.Csv = DummyCsv

    filters = [":temporal"]
    text_analyzer, csv_analyzer = analyzer.get_analyzers(corpus, filters=filters)

    assert isinstance(text_analyzer, DummyText)
    assert isinstance(csv_analyzer, DummyCsv)
    # CSV should be filtered to only row 1
    assert len(csv_analyzer.df) == 1
    assert 1 in csv_analyzer.df.index


def test_get_analyzers_with_temporal_text_filter():
    """Test get_analyzers with temporal:text filter (explicit text→df direction)."""
    corpus = DummyCorpus()
    # Add temporal_links to documents
    corpus.documents[1].metadata["temporal_links"] = [
        {"df_index": 1, "time_gap_seconds": 10}
    ]
    corpus.df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    analyzer.Text = DummyText
    analyzer.Csv = DummyCsv

    filters = ["temporal:text"]
    text_analyzer, csv_analyzer = analyzer.get_analyzers(corpus, filters=filters)

    assert isinstance(text_analyzer, DummyText)
    assert isinstance(csv_analyzer, DummyCsv)
    # CSV should be filtered to only row 1
    assert len(csv_analyzer.df) == 1
    assert 1 in csv_analyzer.df.index


def test_get_analyzers_with_temporal_df_filter():
    """Test get_analyzers with temporal:df filter (df→text direction)."""
    corpus = DummyCorpus()
    corpus.documents[0].id = "doc1"
    corpus.documents[1].id = "doc2"
    # Add temporal_links to documents
    corpus.documents[0].metadata["temporal_links"] = [
        {"df_index": 0, "time_gap_seconds": 5}
    ]
    corpus.documents[1].metadata["temporal_links"] = [
        {"df_index": 2, "time_gap_seconds": 10}
    ]
    corpus.df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]}, index=[0, 1, 2])
    analyzer.Text = DummyText
    analyzer.Csv = DummyCsv

    filters = ["temporal:df"]
    text_analyzer, csv_analyzer = analyzer.get_analyzers(corpus, filters=filters)

    assert isinstance(text_analyzer, DummyText)
    assert isinstance(csv_analyzer, DummyCsv)
    # Documents should be filtered to only those linked from df rows 0 and 2
    assert len(text_analyzer.corpus.documents) == 2
    assert text_analyzer.corpus.documents[0].id == "doc1"
    assert text_analyzer.corpus.documents[1].id == "doc2"


def test_get_analyzers_with_combined_filters():
    """Test get_analyzers with both regular and link filters."""
    corpus = DummyCorpus()
    corpus.documents[0].metadata["embedding_links"] = [
        {"df_index": 0, "similarity": 0.9}
    ]
    corpus.documents[0].metadata["keywords"] = "mask"
    corpus.df = pd.DataFrame(
        {"keywords": ["mask", "vaccine", "mask"], "value": [1, 2, 3]}
    )
    analyzer.Text = DummyText
    analyzer.Csv = DummyCsv

    filters = ["keywords=mask", "=embedding"]
    text_analyzer, csv_analyzer = analyzer.get_analyzers(corpus, filters=filters)

    assert isinstance(text_analyzer, DummyText)
    assert isinstance(csv_analyzer, DummyCsv)


def test_get_analyzers_no_dataframe():
    """Test get_analyzers when corpus has no dataframe."""
    corpus = DummyCorpus()
    corpus.df = None
    analyzer.Text = DummyText
    analyzer.Csv = DummyCsv

    text_analyzer, csv_analyzer = analyzer.get_analyzers(corpus)

    assert isinstance(text_analyzer, DummyText)
    assert csv_analyzer is None


def test_get_analyzers_no_documents():
    """Test get_analyzers when corpus has no documents."""
    corpus = DummyCorpus()
    corpus.documents = []
    corpus.df = pd.DataFrame({"col1": [1, 2, 3]})
    analyzer.Text = DummyText
    analyzer.Csv = DummyCsv

    text_analyzer, csv_analyzer = analyzer.get_analyzers(corpus)

    # Text analyzer should be None when there are no documents
    assert text_analyzer is None
    assert isinstance(csv_analyzer, DummyCsv)


def test__process_csv():
    csv_analyzer = DummyCsv()
    text, ignore = analyzer._process_csv(
        csv_analyzer, "t1", "i1", filters=["x:y", "y=z"]
    )
    assert text == "t1"
    assert ignore == "i1"


def test_get_analyzers_with_id_filter():
    """Test get_analyzers with id filter for synchronized ID linkage."""
    corpus = DummyCorpus()
    corpus.documents[0].id = "doc1"
    corpus.documents[1].id = "doc2"
    corpus.df = pd.DataFrame({"id": ["doc1", "doc2", "doc3"], "col1": [1, 2, 3]})
    analyzer.Text = DummyText
    analyzer.Csv = DummyCsv

    filters = ["id=doc1"]
    text_analyzer, csv_analyzer = analyzer.get_analyzers(corpus, filters=filters)

    assert isinstance(text_analyzer, DummyText)
    assert isinstance(csv_analyzer, DummyCsv)
    # DataFrame should be filtered to only rows where id="doc1"
    assert len(csv_analyzer.df) == 1
    assert csv_analyzer.df.iloc[0]["id"] == "doc1"


def test_get_analyzers_with_id_filter_colon_separator():
    """Test get_analyzers with id filter using colon separator."""
    corpus = DummyCorpus()
    corpus.documents[0].id = "doc1"
    corpus.documents[1].id = "doc2"
    corpus.df = pd.DataFrame({"id": ["doc1", "doc2", "doc3"], "col1": [1, 2, 3]})
    analyzer.Text = DummyText
    analyzer.Csv = DummyCsv

    filters = ["id:doc2"]
    text_analyzer, csv_analyzer = analyzer.get_analyzers(corpus, filters=filters)

    assert isinstance(text_analyzer, DummyText)
    assert isinstance(csv_analyzer, DummyCsv)
    # DataFrame should be filtered to only rows where id="doc2"
    assert len(csv_analyzer.df) == 1
    assert csv_analyzer.df.iloc[0]["id"] == "doc2"


def test_get_analyzers_with_id_filter_no_id_column():
    """Test get_analyzers with id filter when DataFrame has no id column."""
    corpus = DummyCorpus()
    corpus.documents[0].id = "doc1"
    corpus.documents[1].id = "doc2"
    corpus.df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    analyzer.Text = DummyText
    analyzer.Csv = DummyCsv

    filters = ["id=doc1"]
    text_analyzer, csv_analyzer = analyzer.get_analyzers(corpus, filters=filters)

    assert isinstance(text_analyzer, DummyText)
    assert isinstance(csv_analyzer, DummyCsv)
    # DataFrame should remain unchanged since there's no id column
    assert len(csv_analyzer.df) == 3


def test_get_analyzers_with_id_filter_no_value_colon():
    """Test get_analyzers with id: filter (no value) - should sync after other filters."""
    corpus = DummyCorpus()
    corpus.documents[0].id = "doc1"
    corpus.documents[1].id = "doc2"
    corpus.df = pd.DataFrame({"id": ["doc1", "doc2", "doc3"], "col1": [1, 2, 3]})
    analyzer.Text = DummyText
    analyzer.Csv = DummyCsv

    # Apply id: with no value (syncs documents to df)
    filters = ["id:"]
    text_analyzer, csv_analyzer = analyzer.get_analyzers(corpus, filters=filters)

    # After id: sync, only df rows with ids matching document IDs are kept
    assert isinstance(text_analyzer, DummyText)
    assert isinstance(csv_analyzer, DummyCsv)
    # DataFrame should have only 2 rows (doc1 and doc2 match the df ids)
    assert len(csv_analyzer.df) == 2


def test_get_analyzers_with_id_filter_no_value_equals():
    """Test get_analyzers with id= filter (no value) - should sync after other filters."""
    corpus = DummyCorpus()
    corpus.documents[0].id = "doc1"
    corpus.documents[1].id = "doc2"
    corpus.df = pd.DataFrame({"id": ["doc1", "doc2", "doc3"], "col1": [1, 2, 3]})
    analyzer.Text = DummyText
    analyzer.Csv = DummyCsv

    # Apply id= with no value (syncs documents to df)
    filters = ["id="]
    text_analyzer, csv_analyzer = analyzer.get_analyzers(corpus, filters=filters)

    # After id= sync, only df rows with ids matching document IDs are kept
    assert isinstance(text_analyzer, DummyText)
    assert isinstance(csv_analyzer, DummyCsv)
    # DataFrame should have only 2 rows (doc1 and doc2 match the df ids)
    assert len(csv_analyzer.df) == 2
