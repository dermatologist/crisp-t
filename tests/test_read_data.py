import os
import json
import logging
from pathlib import Path

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_corpus_not_none(read_data_fixture):
    corpus = read_data_fixture.create_corpus(
        name="Test Corpus", description="This is a test corpus"
    )
    assert corpus is not None, "Corpus should not be None"


def test_corpus_has_documents(read_data_fixture):
    corpus = read_data_fixture.create_corpus(
        name="Test Corpus", description="This is a test corpus"
    )
    assert len(corpus.documents) > 0, "Corpus should have documents"
    assert all(
        doc is not None for doc in corpus.documents
    ), "All documents should be non-None"

def test_get_document_by_id(read_data_fixture):
    corpus = read_data_fixture.create_corpus(
        name="Test Corpus", description="This is a test corpus"
    )
    first_doc_id = corpus.documents[0].id
    doc = read_data_fixture.get_document_by_id(first_doc_id)
    assert doc is not None, "Document should not be None"
    assert doc.id == first_doc_id, "Document ID should match"

def test_corpus_is_saved_as_json(read_data_fixture):
    corpus = read_data_fixture.create_corpus(
        name="Test Corpus", description="This is a test corpus"
    )
    file_path = str(Path(__file__).parent / "resources" / "")
    read_data_fixture.write_corpus_to_json(file_path)
    assert os.path.exists(file_path), "Corpus JSON file should exist"
    file_name = file_path + "/corpus.json"
    with open(file_name, "r") as f:
        data = json.load(f)
    assert data is not None, "JSON data should not be None"
    assert "documents" in data, "JSON data should contain 'documents' key"
    assert len(data["documents"]) > 0, "'documents' key should have documents"
    # clean up
    # os.remove(file_name)
    # assert not os.path.exists(file_name), "Corpus JSON file should be deleted"
    file_name = file_path + "/corpus_df.csv"
    if os.path.exists(file_name):
        os.remove(file_name)
        assert not os.path.exists(file_name), "Corpus CSV file should be deleted"


def test_corpus_as_dataframe(read_data_fixture):
    corpus = read_data_fixture.create_corpus(
        name="Test Corpus", description="This is a test corpus"
    )
    df = read_data_fixture.corpus_as_dataframe()
    assert df is not None, "DataFrame should not be None"
    assert len(df) > 0, "DataFrame should have rows"


