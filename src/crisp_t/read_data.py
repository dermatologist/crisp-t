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

import pandas as pd
import requests
from pypdf import PdfReader

from .model import Corpus, Document
from .csv import Csv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ReadData:

    def __init__(self, source=None):
        self._content = ""
        self._corpus = None
        self._source = source
        self._documents = []

    @property
    def corpus(self):
        """
        Get the corpus.
        """
        if not self._corpus:
            raise ValueError("No corpus found. Please create a corpus first.")
        return self._corpus

    @property
    def documents(self):
        """
        Get the documents.
        """
        if not self._documents:
            raise ValueError("No documents found. Please read data first.")
        return self._documents

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

    def pretty_print(self):
        """
        Pretty print the corpus.
        """
        if not self._corpus:
            self.create_corpus()
        if self._corpus:
            print(self._corpus.model_dump_json(indent=4))
            logger.info("Corpus: %s", self._corpus.model_dump_json(indent=4))
        else:
            logger.error("No corpus available to pretty print.")

    def create_corpus(self, name=None, description=None):
        """
        Create a corpus from the documents.
        """
        if not self._documents:
            raise ValueError("No documents found. Please read data first.")
        self._corpus = Corpus(
            documents=self._documents,
            df=None,
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

    # write corpus to json file
    def write_corpus_to_json(self, file_name="corpus.json"):
        """
        Write the corpus to a json file.
        """
        if self._source:
            file_name = os.path.join(self._source, file_name)
        if not self._corpus:
            raise ValueError("No corpus found. Please create a corpus first.")
        with open(file_name, "w") as f:
            json.dump(self._corpus.model_dump(), f, indent=4)
        logger.info("Corpus written to %s", file_name)

    # read corpus from json file
    def read_corpus_from_json(self, file_name="corpus.json"):
        """
        Read the corpus from a json file.
        """
        if self._source:
            file_name = os.path.join(self._source, file_name)
        if not os.path.exists(file_name):
            raise ValueError("File not found: %s" % file_name)
        with open(file_name, "r") as f:
            data = json.load(f)
            self._corpus = Corpus.model_validate(data)
            logger.info("Corpus read from %s", file_name)
        return self._corpus

    def read_csv(
        self,
        file_name,
        comma_separated_ignore_words=None,
        comma_separated_text_columns="",
        id_column="",
        numeric=False,
    ):
        """
        Read the corpus from a csv file.
        """
        if not os.path.exists(file_name):
            raise ValueError("File not found: %s" % file_name)
        df = pd.read_csv(file_name)
        if numeric:
            text_columns = comma_separated_text_columns.split(",")
            # remove text columns from the dataframe
            for column in text_columns:
                if column in df.columns:
                    df.drop(column, axis=1, inplace=True)
            return df
        if comma_separated_text_columns:
            text_columns = comma_separated_text_columns.split(",")
        else:
            text_columns = df.columns.tolist()
        for index, row in df.iterrows():
            read_from_file = ""
            for column in text_columns:
                read_from_file += str(row[column]) + " "
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
                    "source": file_name,
                    "file_name": file_name,
                    "row": index,
                    "id": row[id_column] if id_column is not "" else "",
                },
                id=str(index),
                score=0.0,
                name="",
                description="",
            )
            self._documents.append(_document)
        logger.info("Corpus read from %s", file_name)

    def read_source(self, source, comma_separated_ignore_words=None):
        if source.endswith("/"):
            self._source = source
            logger.info("Reading data from folder: %s", source)
            for file_name in os.listdir(source):
                if file_name.endswith(".txt"):
                    with open(os.path.join(source, file_name), "r") as f:
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
                                "source": os.path.join(source, file_name),
                                "file_name": file_name,
                            },
                            id=file_name,
                            score=0.0,
                            name="",
                            description="",
                        )
                        self._documents.append(_document)
                if file_name.endswith(".pdf"):
                    with open(os.path.join(source, file_name), "rb") as f:
                        reader = PdfReader(f)
                        read_from_file = ""
                        for page in reader.pages:
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
                                "source": os.path.join(source, file_name),
                                "file_name": file_name,
                            },
                            id=file_name,
                            score=0.0,
                            name="",
                            description="",
                        )
                        self._documents.append(_document)
        # if source is a url
        elif source.startswith("http://") or source.startswith("https://"):
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
        else:
            raise ValueError("source must be a folder path or url.")

        """
        Combine duplicate topics using Dict
        """

    def corpus_as_dataframe(self):
        """
        Convert the corpus to a pandas dataframe.
        """
        if not self._corpus:
            raise ValueError("No corpus found. Please create a corpus first.")
        data = []
        for document in self._corpus.documents:
            data.append(document.model_dump())
        df = pd.DataFrame(data)
        return df
