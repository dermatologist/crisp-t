import types

import pytest

from src.crisp_t.helpers import analyzer, initializer


class DummyCorpus:
    def __init__(self):
        self.documents = ["doc1", "doc2"]
        self.df = None


class DummyCsv:
    def __init__(self, corpus=None):
        self.corpus = corpus
        self.comma_separated_text_columns = ""
        self.comma_separated_ignore_columns = ""
        self.df = None

    def get_shape(self):
        return (2, 2)

    def filter_rows_by_column_value(self, key, value):
        return None


class DummyText:
    def __init__(self, corpus=None):
        self.corpus = corpus

    def filter_documents(self, key, value):
        return None

    def document_count(self):
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
    corpus.df = True
    analyzer.Csv = DummyCsv
    csv_analyzer = analyzer.get_csv_analyzer(
        corpus, "text_col", "ignore_col", filters=["a=b"]
    )
    assert isinstance(csv_analyzer, DummyCsv)
    assert csv_analyzer.comma_separated_text_columns == "text_col"
    assert csv_analyzer.comma_separated_ignore_columns == "ignore_col"


def test__process_csv():
    csv_analyzer = DummyCsv()
    text, ignore = analyzer._process_csv(
        csv_analyzer, "t1", "i1", filters=["x:y", "y=z"]
    )
    assert text == "t1"
    assert ignore == "i1"
