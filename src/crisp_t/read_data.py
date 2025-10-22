"""
Copyright (C) 2025 Bell Eapen

This file is part of crisp-t.

crisp-t is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

crisp-t is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with crisp-t.  If not, see <https://www.gnu.org/licenses/>.
"""

import json
import logging
import os
import re
import tempfile
import pandas as pd
import requests
from pypdf import PdfReader
from tqdm import tqdm

from .csv import Csv
from .model import Corpus, Document

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ReadData:

    def __init__(self, corpus: Corpus | None = None, source=None):
        self._content = ""
        self._corpus = corpus
        self._source = source
        self._documents = []
        self._df = pd.DataFrame()

    @property
    def corpus(self):
        """
        Get the corpus.
        """
        if not self._corpus:
            raise ValueError("No corpus found. Please create a corpus first.")
        self._corpus.documents = self._documents
        self._corpus.df = self._df
        return self._corpus

    @property
    def documents(self):
        """
        Get the documents.
        """
        if not self._documents:
            raise ValueError("No documents found. Please read data first.")
        return self._documents

    @property
    def df(self):
        """
        Get the dataframe.
        """
        if self._df is None:
            raise ValueError("No dataframe found. Please read data first.")
        return self._df

    @corpus.setter
    def corpus(self, value):
        """
        Set the corpus.
        """
        if not isinstance(value, Corpus):
            raise ValueError("Value must be a Corpus object.")
        self._corpus = value

    @documents.setter
    def documents(self, value):
        """
        Set the documents.
        """
        if not isinstance(value, list):
            raise ValueError("Value must be a list of Document objects.")
        for document in value:
            if not isinstance(document, Document):
                raise ValueError("Value must be a list of Document objects.")
        self._documents = value

    @df.setter
    def df(self, value):
        """
        Set the dataframe.
        """
        if not isinstance(value, pd.DataFrame):
            raise ValueError("Value must be a pandas DataFrame.")
        self._df = value

    def pretty_print(self):
        """
        Pretty print the corpus.
        """
        if not self._corpus:
            self.create_corpus()
        if self._corpus:
            print(
                self._corpus.model_dump_json(indent=4, exclude={"df", "visualization"})
            )
            logger.info(
                "Corpus: %s",
                self._corpus.model_dump_json(indent=4, exclude={"df", "visualization"}),
            )
        else:
            logger.error("No corpus available to pretty print.")

    # TODO: Enforce only one corpus (Singleton pattern)
    def create_corpus(self, name=None, description=None):
        """
        Create a corpus from the documents and dataframe.
        """
        if not self._documents:
            raise ValueError("No documents found. Please read data first.")
        if self._corpus:
            self._corpus.documents = self._documents
            self._corpus.df = self._df
        else:
            self._corpus = Corpus(
                documents=self._documents,
                df=self._df,
                visualization={},
                metadata={},
                id="corpus",
                score=0.0,
                name=name,
                description=description,
            )
        return self._corpus

    def get_documents_from_corpus(self):
        """
        Get the documents from the corpus.
        """
        if not self._corpus:
            raise ValueError("No corpus found. Please create a corpus first.")
        return self._corpus.documents

    def get_document_by_id(self, doc_id):
        """
        Get a document from the corpus by its ID.
        """
        if not self._corpus:
            raise ValueError("No corpus found. Please create a corpus first.")
        for document in tqdm(self._corpus.documents, desc="Searching documents", disable=len(self._corpus.documents) < 10):
            if document.id == doc_id:
                return document
        raise ValueError("Document not found: %s" % doc_id)

    def write_corpus_to_json(self, file_path="", corpus=None):
        """
        Write the corpus to a json file.

        Accepts either a directory path or an explicit file path ending with
        'corpus.json'. In both cases, a sibling 'corpus_df.csv' will be written
        next to the json if a DataFrame is available.
        """
        from pathlib import Path

        path = Path(file_path)
        # Determine targets
        if path.suffix:  # treat as explicit file path
            file_name = path
            df_name = path.with_name("corpus_df.csv")
        else:
            file_name = path / "corpus.json"
            df_name = path / "corpus_df.csv"

        corp = corpus if corpus is not None else self._corpus
        if not corp:
            raise ValueError("No corpus found. Please create a corpus first.")
        file_name.parent.mkdir(parents=True, exist_ok=True)
        with open(file_name, "w") as f:
            json.dump(corp.model_dump(exclude={"df", "visualization"}), f, indent=4)
        if getattr(corp, "df", None) is not None and not corp.df.empty:
            corp.df.to_csv(df_name, index=False)
        logger.info("Corpus written to %s", file_name)

    def read_corpus_from_json(self, file_path="", comma_separated_ignore_words=""):
        """
        Read the corpus from a json file.
        """
        from pathlib import Path

        file_path = Path(file_path)
        file_name = file_path / "corpus.json"
        df_name = file_path / "corpus_df.csv"
        if self._source:
            file_name = Path(self._source) / file_name
        if not file_name.exists():
            raise ValueError(f"File not found: {file_name}")
        with open(file_name, "r") as f:
            data = json.load(f)
            self._corpus = Corpus.model_validate(data)
            logger.info(f"Corpus read from {file_name}")
        if df_name.exists():
            self._corpus.df = pd.read_csv(df_name)
        else:
            self._corpus.df = None
        # Remove ignore words from self._corpus.documents text
        documents = []
        for document in tqdm(self._corpus.documents, desc="Processing documents", disable=len(self._corpus.documents) < 10):
            if comma_separated_ignore_words:
                for word in comma_separated_ignore_words.split(","):
                    document.text = re.sub(
                        r"\b" + word.strip() + r"\b",
                        "",
                        document.text,
                        flags=re.IGNORECASE,
                    )
            documents.append(document)
        self._corpus.documents = documents
        return self._corpus

    def read_csv_to_corpus(
        self,
        file_name,
        comma_separated_ignore_words=None,
        comma_separated_text_columns="",
        id_column="",
    ):
        """
        Read the corpus from a csv file.
        """
        from pathlib import Path

        file_name = Path(file_name)
        if not file_name.exists():
            raise ValueError(f"File not found: {file_name}")
        df = pd.read_csv(file_name)
        original_df = df.copy()
        if comma_separated_text_columns:
            text_columns = comma_separated_text_columns.split(",")
        else:
            text_columns = []
        # remove text columns from the dataframe
        for column in text_columns:
            if column in df.columns:
                df.drop(column, axis=1, inplace=True)
        # Set self._df to the numeric part after dropping text columns
        self._df = df.copy()
        for index, row in tqdm(original_df.iterrows(), total=len(original_df), desc="Reading CSV rows", disable=len(original_df) < 10):
            read_from_file = ""
            for column in text_columns:
                read_from_file += f"{row[column]} "
            # remove comma separated ignore words
            if comma_separated_ignore_words:
                for word in comma_separated_ignore_words.split(","):
                    read_from_file = re.sub(
                        r"\b" + word.strip() + r"\b",
                        "",
                        read_from_file,
                        flags=re.IGNORECASE,
                    )
            self._content += read_from_file
            _document = Document(
                text=read_from_file,
                metadata={
                    "source": str(file_name),
                    "file_name": str(file_name),
                    "row": index,
                    "id": (
                        row[id_column]
                        if (id_column != "" and id_column in original_df.columns)
                        else index
                    ),
                },
                id=str(index),
                score=0.0,
                name="",
                description="",
            )
            self._documents.append(_document)
        logger.info(f"Corpus read from {file_name}")
        self.create_corpus()
        return self._corpus

    def read_source(
        self, source, comma_separated_ignore_words=None, comma_separated_text_columns=""
    ):
        _CSV_EXISTS = False
        # if source is a url
        if source.startswith("http://") or source.startswith("https://"):
            response = requests.get(source)
            if response.status_code == 200:
                read_from_file = response.text
                # remove comma separated ignore words
                if comma_separated_ignore_words:
                    for word in comma_separated_ignore_words.split(","):
                        read_from_file = re.sub(
                            r"\b" + word.strip() + r"\b",
                            "",
                            read_from_file,
                            flags=re.IGNORECASE,
                        )
                self._content = read_from_file
                _document = Document(
                    text=read_from_file,
                    metadata={"source": source},
                    id=source,
                    score=0.0,
                    name="",
                    description="",
                )
                self._documents.append(_document)
        elif os.path.exists(source):
            from pathlib import Path

            source_path = Path(source)
            self._source = source
            logger.info(f"Reading data from folder: {source}")
            file_list = os.listdir(source)
            for file_name in tqdm(file_list, desc="Reading files", disable=len(file_list) < 10):
                file_path = source_path / file_name
                if file_name.endswith(".txt"):
                    with open(file_path, "r") as f:
                        read_from_file = f.read()
                        # remove comma separated ignore words
                        if comma_separated_ignore_words:
                            for word in comma_separated_ignore_words.split(","):
                                read_from_file = re.sub(
                                    r"\b" + word.strip() + r"\b",
                                    "",
                                    read_from_file,
                                    flags=re.IGNORECASE,
                                )
                        self._content += read_from_file
                        _document = Document(
                            text=read_from_file,
                            metadata={
                                "source": str(file_path),
                                "file_name": file_name,
                            },
                            id=file_name,
                            score=0.0,
                            name="",
                            description="",
                        )
                        self._documents.append(_document)
                if file_name.endswith(".pdf"):
                    with open(file_path, "rb") as f:
                        reader = PdfReader(f)
                        read_from_file = ""
                        for page in tqdm(reader.pages, desc=f"Reading PDF {file_name}", leave=False, disable=len(reader.pages) < 10):
                            read_from_file += page.extract_text()
                        # remove comma separated ignore words
                        if comma_separated_ignore_words:
                            for word in comma_separated_ignore_words.split(","):
                                read_from_file = re.sub(
                                    r"\b" + word.strip() + r"\b",
                                    "",
                                    read_from_file,
                                    flags=re.IGNORECASE,
                                )
                        self._content += read_from_file
                        _document = Document(
                            text=read_from_file,
                            metadata={
                                "source": str(file_path),
                                "file_name": file_name,
                            },
                            id=file_name,
                            score=0.0,
                            name="",
                            description="",
                        )
                        self._documents.append(_document)
                if file_name.endswith(".csv") and comma_separated_text_columns == "":
                    logger.info(f"Reading CSV file: {file_path}")
                    self._df = Csv().read_csv(file_path)
                    logger.info(f"CSV file read with shape: {self._df.shape}")
                    _CSV_EXISTS = True
                if file_name.endswith(".csv") and comma_separated_text_columns != "":
                    logger.info(f"Reading CSV file to corpus: {file_path}")
                    self.read_csv_to_corpus(
                        file_path,
                        comma_separated_ignore_words,
                        comma_separated_text_columns,
                    )
                    logger.info(
                        f"CSV file read to corpus with documents: {len(self._documents)}"
                    )
                    _CSV_EXISTS = True
            if not _CSV_EXISTS:
                # create a simple csv with columns: id, number, text
                # and fill it with random data
                _csv = """
id,number,response
1,100,Sample text one
2,200,Sample text two
3,300,Sample text three
4,400,Sample text four
"""
                # write the csv to a temp file
                with tempfile.NamedTemporaryFile(
                    mode="w+", delete=False, suffix=".csv"
                ) as temp_csv:
                    temp_csv.write(_csv)
                    temp_csv_path = temp_csv.name
                logger.info(f"No CSV found. Created temp CSV file: {temp_csv_path}")
                self._df = Csv().read_csv(temp_csv_path)
                logger.info(f"CSV file read with shape: {self._df.shape}")
                # remove the temp file
                os.remove(temp_csv_path)


        else:
            raise ValueError(f"Source not found: {source}")

    def corpus_as_dataframe(self):
        """
        Convert the corpus to a pandas dataframe.
        """
        if not self._corpus:
            raise ValueError("No corpus found. Please create a corpus first.")
        data = []
        for document in tqdm(self._corpus.documents, desc="Converting to dataframe", disable=len(self._corpus.documents) < 10):
            data.append(document.model_dump())
        df = pd.DataFrame(data)
        return df
