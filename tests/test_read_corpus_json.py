import os
import re

import pytest

# Assumes pytest fixture 'read_data_fixture' is available and corpus.json exists in resources


def test_read_corpus_from_json_ignore_words(read_data_fixture):
    resources_dir = os.path.join(os.path.dirname(__file__), "resources")
    # Use two ignore words that should be present in the corpus.json documents
    ignore_words = "the,and"
    corpus = read_data_fixture.read_corpus_from_json(
        resources_dir, comma_separated_ignore_words=ignore_words
    )
    assert corpus is not None, "Corpus should not be None"
    assert hasattr(corpus, "documents"), "Corpus should have documents attribute"
    assert len(corpus.documents) > 0, "Corpus should have at least one document"
    # Check that ignore words are removed from all document texts
    for doc in corpus.documents:
        for word in ignore_words.split(","):
            # Should not find the word as a whole word (case-insensitive)
            assert not re.search(
                r"\b" + re.escape(word) + r"\b", doc.text, re.IGNORECASE
            ), f"Ignore word '{word}' found in document text: {doc.text}"
