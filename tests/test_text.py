import pytest
import os
import json
import logging
from src.crisp_t.read_data import ReadData
from src.crisp_t.text import Text
from pkg_resources import resource_filename

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def document_fixture():
    folder_path = resource_filename("src.crisp_t.resources", "/")
    read_data = ReadData()
    read_data.read_source(folder_path)
    corpus = read_data.create_corpus(name="Test Corpus", description="This is a test corpus")
    document = corpus.documents[0]
    return document


def test_text_initialization(document_fixture):
    text = Text(document=document_fixture)
    assert text.document == document_fixture, "Document should be set correctly"


def test_common_words(document_fixture):
    text = Text(document=document_fixture)
    text.make_spacy_doc()
    common_words = text.common_words(index=5)
    print(
        "Common words:", common_words
    )
    # Check if the common words are returned as expected
    # [('theory', 4), ('evaluation', 2), ('glaser', 1), ('classical', 1), ('number', 1)]
    assert isinstance(common_words, list), "Common words should be a list"
    assert len(common_words) == 5, "Common words should contain 5 items"
    assert all(isinstance(item, tuple) and len(item) == 2 for item in common_words), "Each item should be a tuple of (word, count)"
    assert all(isinstance(item[0], str) and isinstance(item[1], int) for item in common_words), "Each tuple should contain a string and an integer"

def test_common_nouns(document_fixture):
    text = Text(document=document_fixture)
    text.make_spacy_doc()
    common_nouns = text.common_nouns(index=5)
    print(
        "Common nouns:", common_nouns
    )
    # Check if the common nouns are returned as expected
    # [('theory', 4), ('evaluation', 2), ('number', 1), ('guideline', 1), ('methodology', 1)]
    assert isinstance(common_nouns, list), "Common nouns should be a list"
    assert len(common_nouns) == 5, "Common nouns should contain 5 items"
    assert all(isinstance(item, tuple) and len(item) == 2 for item in common_nouns), "Each item should be a tuple of (word, count)"
    assert all(isinstance(item[0], str) and isinstance(item[1], int) for item in common_nouns), "Each tuple should contain a string and an integer"

def test_common_verbs(document_fixture):
    text = Text(document=document_fixture)
    text.make_spacy_doc()
    common_verbs = text.common_verbs(index=3)
    print(
        "Common verbs:", common_verbs
    )
    # Check if the common verbs are returned as expected
    # [('theory', 4), ('evaluation', 2), ('number', 1), ('guideline', 1), ('methodology', 1)]
    assert isinstance(common_verbs, list), "Common verbs should be a list"
    assert len(common_verbs) == 3, "Common verbs should contain 3 items"
    assert all(isinstance(item, tuple) and len(item) == 2 for item in common_verbs), "Each item should be a tuple of (word, count)"
    assert all(isinstance(item[0], str) and isinstance(item[1], int) for item in common_verbs), "Each tuple should contain a string and an integer"

def test_sentences_with_common_nouns(document_fixture):
    text = Text(document=document_fixture)
    text.make_spacy_doc()
    sentences_with_common_nouns = text.sentences_with_common_nouns(index=5)
    print(
        "Sentences with common nouns:", sentences_with_common_nouns
    )
    # Check if the sentences with common nouns are returned as expected
    assert isinstance(sentences_with_common_nouns, list), "Sentences with common nouns should be a list"
    assert len(sentences_with_common_nouns) > 0, "Sentences with common nouns should contain at least one item"
    assert all(isinstance(item, str) for item in sentences_with_common_nouns), "Each item should be a string"

def test_spans_with_common_nouns(document_fixture):
    text = Text(document=document_fixture)
    text.make_spacy_doc()
    spans_with_common_nouns = text.spans_with_common_nouns(word="evaluation")
    print(
        "Spans with common nouns:", spans_with_common_nouns
    )
    # Check if the spans with common nouns are returned as expected
    assert isinstance(spans_with_common_nouns, list), "Spans with common nouns should be a list"


def test_dimensions(document_fixture):
    text = Text(document=document_fixture)
    text.make_spacy_doc()
    dimensions = text.dimensions(word="theory", index=3)
    print(
        "Dimensions:", dimensions
    )
    # Check if the dimensions are returned as expected
    assert isinstance(dimensions, list), "Dimensions should be a list"
    assert len(dimensions) == 3, "Dimensions should contain 3 items"
    assert all(isinstance(item, tuple) and len(item) == 2 for item in dimensions), "Each item should be a tuple of (word, count)"
    assert all(isinstance(item[0], str) and isinstance(item[1], int) for item in dimensions), "Each tuple should contain a string and an integer"
