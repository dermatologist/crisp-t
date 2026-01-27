import json
import logging
import os
from pathlib import Path

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.crisp_t.read_data import extract_timestamp_from_text


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


def test_extract_timestamp_iso_8601():
    """Test extraction of ISO 8601 timestamp format."""
    text = "This document was created on 2025-01-15T10:30:00Z."
    timestamp = extract_timestamp_from_text(text)
    assert timestamp is not None, "Should extract ISO 8601 timestamp"
    assert "2025-01-15" in timestamp, "Extracted timestamp should contain date"


def test_extract_timestamp_date_only():
    """Test extraction of date-only format."""
    text = "Meeting on 2025-01-15 was very productive."
    timestamp = extract_timestamp_from_text(text)
    assert timestamp is not None, "Should extract date-only timestamp"
    assert "2025-01-15" in timestamp, "Extracted timestamp should contain date"


def test_extract_timestamp_common_format():
    """Test extraction of common date format (MM/DD/YYYY)."""
    text = "Interview conducted on 01/15/2025."
    timestamp = extract_timestamp_from_text(text)
    assert timestamp is not None, "Should extract common date format"


def test_extract_timestamp_none():
    """Test that None is returned when no timestamp found."""
    text = "This is just plain text without any dates."
    timestamp = extract_timestamp_from_text(text)
    assert timestamp is None, "Should return None when no timestamp found"


def test_extract_timestamp_empty_string():
    """Test that None is returned for empty string."""
    timestamp = extract_timestamp_from_text("")
    assert timestamp is None, "Should return None for empty string"


def test_txt_file_with_timestamp():
    """Test that timestamps are extracted from txt files."""
    from src.crisp_t.read_data import ReadData

    folder_path = str(Path(__file__).parent / "resources" / "")
    read_data = ReadData()
    read_data.read_source(folder_path)
    corpus = read_data.create_corpus()

    # Find document from interview-with-date.txt
    doc_with_timestamp = None
    for doc in corpus.documents:
        if "interview-with-date.txt" in str(doc.metadata.get("file_name", "")):
            doc_with_timestamp = doc
            break

    assert (
        doc_with_timestamp is not None
    ), "Should find interview-with-date.txt document"
    assert (
        doc_with_timestamp.timestamp is not None
    ), "Document should have extracted timestamp"
    assert (
        "2025-01-15" in doc_with_timestamp.timestamp
    ), "Timestamp should be from file content"
