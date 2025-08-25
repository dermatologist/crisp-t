import os
import json
import logging
import pathlib
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
    # get folder path to src.crisp_t.resources
    file_path = pathlib.Path("src/crisp_t/resources/")
    read_data_fixture.write_corpus_to_json(file_path)
    assert os.path.exists(file_path), "Corpus JSON file should exist"
    with open(file_path / "corpus.json", "r") as f:
        data = json.load(f)
    assert data is not None, "JSON data should not be None"
    assert "documents" in data, "JSON data should contain 'documents' key"
    assert len(data["documents"]) > 0, "'documents' key should have documents"
    # clean up
    os.remove(file_path / "corpus.json")
    assert not os.path.exists(file_path / "corpus.json"), "Corpus JSON file should be deleted"


def test_corpus_as_dataframe(read_data_fixture):
    corpus = read_data_fixture.create_corpus(
        name="Test Corpus", description="This is a test corpus"
    )
    df = read_data_fixture.corpus_as_dataframe()
    assert df is not None, "DataFrame should not be None"
    assert len(df) > 0, "DataFrame should have rows"


def test_read_csv(read_data_fixture):
    folder_path = pathlib.Path("src/crisp_t/resources/food_coded.csv")
    read_data_fixture.read_csv(
        folder_path,
        comma_separated_text_columns="comfort_food,comfort_food_reasons,diet_current",
    )
    assert len(read_data_fixture.documents) > 0, "Documents should be read from CSV"
    assert all(
        doc is not None for doc in read_data_fixture.documents
    ), "All documents should be non-None"
    read_data_fixture.pretty_print()


def test_read_csv_numeric(read_data_fixture):
    folder_path = pathlib.Path("src/crisp_t/resources/food_coded.csv")
    df = read_data_fixture.read_csv(
        folder_path,
        comma_separated_text_columns="comfort_food,comfort_food_reasons,diet_current",
        numeric=True,
    )
    print(df.head())
    assert df is not None, "DataFrame should not be None"
    assert len(df) > 0, "DataFrame should have rows"
