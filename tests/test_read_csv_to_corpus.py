import os
import tempfile

import pandas as pd

from src.crisp_t.read_data import ReadData


def test_read_csv_to_corpus_creates_documents_and_df():
    # Create a temporary CSV file
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "text_col": ["This is doc one.", "This is doc two."],
                "value": [10, 20],
            }
        )
        df.to_csv(csv_path, index=False)

        # Read CSV to corpus
        reader = ReadData()
        corpus = reader.read_csv_to_corpus(
            csv_path,
            comma_separated_ignore_words=None,
            comma_separated_text_columns="text_col",
            id_column="id",
        )

        # Check corpus and documents
        assert corpus is not None, "Corpus should not be None"
        assert len(corpus.documents) == 2, "Should create one document per row"
        assert all(
            doc.text.strip() != "" for doc in corpus.documents
        ), "Document text should not be empty"
        # Check that DataFrame is set
        assert isinstance(reader.df, pd.DataFrame), "DataFrame should be set"
        assert reader.df.shape[0] == 2, "DataFrame should have two rows"
        # Check document IDs
        assert corpus.documents[0].metadata["id"] == 1
        assert corpus.documents[1].metadata["id"] == 2
