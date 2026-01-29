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

import datetime
import json
import logging
import multiprocessing
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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


def extract_timestamp_from_text(text: str) -> str | None:
    """
    Extract the first occurrence of a timestamp from text.
    Supports ISO 8601 format and common date formats.

    Args:
        text: Text to search for timestamps

    Returns:
        ISO 8601 formatted timestamp string or None if not found
    """
    if not text:
        return None

    # Pattern for ISO 8601: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS or variants
    iso_pattern = (
        r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d{3})?(?:Z|[+-]\d{2}:\d{2})?)?"
    )
    iso_match = re.search(iso_pattern, text)
    if iso_match:
        timestamp_str = iso_match.group(0)
        try:
            # Parse and return as ISO 8601 string
            dt = pd.to_datetime(timestamp_str)
            return dt.isoformat()
        except Exception:
            pass

    # Pattern for common date formats: MM/DD/YYYY or DD/MM/YYYY
    common_pattern = r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}"
    common_match = re.search(common_pattern, text)
    if common_match:
        timestamp_str = common_match.group(0)
        try:
            dt = pd.to_datetime(timestamp_str)
            return dt.isoformat()
        except Exception:
            pass

    return None


def extract_tag_from_filename(filename: str) -> str | None:
    """
    Extract a tag from a filename by splitting on - or _ and taking the first part.

    Args:
        filename: The filename to extract tag from (e.g., "interview-1.txt")

    Returns:
        The tag (first part before - or _) or None if no separator found.
        The tag is the part before the first - or _ character.
        Example: "interview-1.txt" -> "interview", "report_2025.pdf" -> "report"
    """
    if not filename:
        return None

    # Remove file extension
    name_without_ext = Path(filename).stem

    # Split on - or _ and get the first part
    if "-" in name_without_ext:
        tag = name_without_ext.split("-")[0]
        return tag if tag else None
    elif "_" in name_without_ext:
        tag = name_without_ext.split("_")[0]
        return tag if tag else None

    return None


class ReadData:

    def __init__(self, corpus: Corpus | None = None, source=None):
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
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            corpus_id = f"corpus_{timestamp}"
            self._corpus = Corpus(
                documents=self._documents,
                df=self._df,
                visualization={},
                metadata={},
                id=corpus_id,
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
        Get a document from the corpus by its ID. Uses parallel search for large corpora.
        """
        if not self._corpus:
            raise ValueError("No corpus found. Please create a corpus first.")
        documents = self._corpus.documents
        if len(documents) < 10:
            for document in tqdm(documents, desc="Searching documents", disable=True):
                if document.id == doc_id:
                    return document
        else:
            n_cores = multiprocessing.cpu_count()
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(lambda doc: doc.id == doc_id, document): i
                    for i, document in enumerate(documents)
                }
                with tqdm(
                    total=len(futures),
                    desc=f"Searching documents (parallel, {n_cores} cores)",
                ) as pbar:
                    for future in as_completed(futures):
                        i = futures[future]
                        found = future.result()
                        pbar.update(1)
                        if found:
                            return documents[i]
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
        if (
            corp.df is not None
            and isinstance(corp.df, pd.DataFrame)
            and not corp.df.empty
        ):
            corp.df.to_csv(df_name, index=False)
        logger.info("Corpus written to %s", file_name)

    # @lru_cache(maxsize=3)
    def read_corpus_from_json(self, file_path="", comma_separated_ignore_words=""):
        """
        Read the corpus from a json file. Parallelizes ignore word removal for large corpora.
        """
        from pathlib import Path

        file_path = Path(file_path)
        file_name = file_path / "corpus.json"
        df_name = file_path / "corpus_df.csv"
        if self._source:
            file_name = Path(self._source) / file_name
        if not file_name.exists():
            raise ValueError(f"File not found: {file_name}")
        with open(file_name) as f:
            data = json.load(f)
            self._corpus = Corpus.model_validate(data)
            logger.info(f"Corpus read from {file_name}")
        if df_name.exists():
            self._corpus.df = pd.read_csv(df_name)
        else:
            self._corpus.df = None
        # Remove ignore words from self._corpus.documents text
        documents = self._corpus.documents

        # Pre-compile regex patterns once for efficiency instead of inside loops
        compiled_patterns = []
        if comma_separated_ignore_words:
            for word in comma_separated_ignore_words.split(","):
                pattern = re.compile(r"\b" + word.strip() + r"\b", flags=re.IGNORECASE)
                compiled_patterns.append(pattern)

        if len(documents) < 10:
            processed_docs = []
            for document in tqdm(documents, desc="Processing documents", disable=True):
                for pattern in compiled_patterns:
                    document.text = pattern.sub("", document.text)
                processed_docs.append(document)
        else:

            def process_doc(document):
                for pattern in compiled_patterns:
                    document.text = pattern.sub("", document.text)
                return document

            processed_docs = []
            n_cores = multiprocessing.cpu_count()
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(process_doc, document): document
                    for document in documents
                }
                with tqdm(
                    total=len(futures),
                    desc=f"Processing documents (parallel, {n_cores} cores)",
                ) as pbar:
                    for future in as_completed(futures):
                        processed_docs.append(future.result())
                        pbar.update(1)
        self._corpus.documents = processed_docs
        return self._corpus

    # @lru_cache(maxsize=3)
    def read_csv_to_corpus(
        self,
        file_name,
        comma_separated_ignore_words=None,
        comma_separated_text_columns="",
        id_column="",
        timestamp_column="",
        max_rows=None,
    ):
        """
        Read the corpus from a csv file. Parallelizes document creation for large CSVs.

        Args:
            file_name: Path to CSV file
            comma_separated_ignore_words: Stop words to ignore
            comma_separated_text_columns: Text columns to extract
            id_column: Column to use as document ID
            timestamp_column: Column containing timestamps (auto-detected if not specified)
            max_rows: Maximum number of rows to read from CSV
        """
        from pathlib import Path

        file_name = Path(file_name)
        if not file_name.exists():
            raise ValueError(f"File not found: {file_name}")

        # Read CSV with optional row limit
        if max_rows is not None:
            df = pd.read_csv(file_name, nrows=max_rows)
            logger.info(f"Limited CSV reading to first {max_rows} rows")
        else:
            df = pd.read_csv(file_name)
        original_df = df.copy()

        # Auto-detect timestamp column if not specified
        if not timestamp_column:
            timestamp_candidates = [
                "timestamp",
                "datetime",
                "time",
                "date",
                "created_at",
                "updated_at",
            ]
            for candidate in timestamp_candidates:
                if candidate in df.columns:
                    timestamp_column = candidate
                    logger.info(f"Auto-detected timestamp column: {timestamp_column}")
                    break

        # Parse timestamp column to datetime if it exists
        if timestamp_column and timestamp_column in df.columns:
            try:
                df[timestamp_column] = pd.to_datetime(df[timestamp_column])
                logger.info(f"Parsed timestamp column '{timestamp_column}' to datetime")
            except Exception as e:
                logger.warning(
                    f"Could not parse timestamp column '{timestamp_column}': {e}"
                )

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
        rows = list(original_df.iterrows())

        # Pre-compile regex patterns once for efficiency instead of inside loops
        compiled_patterns = []
        if comma_separated_ignore_words:
            for word in comma_separated_ignore_words.split(","):
                pattern = re.compile(r"\b" + word.strip() + r"\b", flags=re.IGNORECASE)
                compiled_patterns.append(pattern)

        def create_document(args):
            index, row = args
            # Use list and join for efficient string concatenation, handle None values
            text_parts = [
                (
                    str(row[column])
                    if row[column] is not None
                    and not (
                        isinstance(row[column], float) and row[column] != row[column]
                    )
                    else ""
                )
                for column in text_columns
            ]
            read_from_file = " ".join(text_parts)
            # Apply pre-compiled patterns
            for pattern in compiled_patterns:
                read_from_file = pattern.sub("", read_from_file)

            # Extract timestamp if available
            doc_timestamp = None
            if timestamp_column and timestamp_column in original_df.columns:
                ts_value = row[timestamp_column]
                if ts_value is not None and not pd.isna(ts_value):
                    # Convert to ISO 8601 string
                    try:
                        if isinstance(ts_value, pd.Timestamp):
                            doc_timestamp = ts_value.isoformat()
                        else:
                            doc_timestamp = pd.to_datetime(ts_value).isoformat()
                    except Exception as exc:
                        # Intentionally ignore timestamp parsing errors and fall back to no timestamp,
                        # but log them for diagnostic purposes.
                        logging.debug(
                            "Failed to parse timestamp value %r in column %r: %s",
                            ts_value,
                            timestamp_column,
                            exc,
                        )

            _document = Document(
                text=read_from_file,
                timestamp=doc_timestamp,
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
            return read_from_file, _document

        if len(rows) < 10:
            results = [
                create_document(args)
                for args in tqdm(rows, desc="Reading CSV rows", disable=True)
            ]
        else:

            results = []
            # import multiprocessing

            n_cores = multiprocessing.cpu_count()
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(create_document, args): args for args in rows
                }
                with tqdm(
                    total=len(futures),
                    desc=f"Reading CSV rows (parallel, {n_cores} cores)",
                ) as pbar:
                    for future in as_completed(futures):
                        results.append(future.result())
                        pbar.update(1)

        if len(results) < 10:
            for read_from_file, _document in tqdm(
                results, desc="Finalizing corpus", disable=True
            ):
                self._documents.append(_document)
        else:

            # import multiprocessing

            n_cores = multiprocessing.cpu_count()
            with tqdm(
                results,
                total=len(results),
                desc=f"Finalizing corpus (parallel, {n_cores} cores)",
            ) as pbar:
                for read_from_file, _document in pbar:
                    self._documents.append(_document)
        logger.info(f"Corpus read from {file_name}")
        self.create_corpus()
        return self._corpus

    def read_source(
        self,
        source,
        comma_separated_ignore_words=None,
        comma_separated_text_columns="",
        max_text_files=None,
        max_csv_rows=None,
    ):
        _CSV_EXISTS = False

        # Pre-compile regex patterns once for efficiency instead of inside loops
        compiled_patterns = []
        if comma_separated_ignore_words:
            for word in comma_separated_ignore_words.split(","):
                pattern = re.compile(r"\b" + word.strip() + r"\b", flags=re.IGNORECASE)
                compiled_patterns.append(pattern)

        def apply_ignore_patterns(text):
            """Apply pre-compiled ignore patterns to text."""
            for pattern in compiled_patterns:
                text = pattern.sub("", text)
            return text

        # if source is a url
        if source.startswith("http://") or source.startswith("https://"):
            response = requests.get(source)
            if response.status_code == 200:
                read_from_file = response.text
                read_from_file = apply_ignore_patterns(read_from_file)
                # Extract timestamp from content
                doc_timestamp = extract_timestamp_from_text(read_from_file)
                # self._content removed
                _document = Document(
                    text=read_from_file,
                    timestamp=doc_timestamp,
                    metadata={"source": source},
                    id=source,
                    score=0.0,
                    name="",
                    description="",
                )
                self._documents.append(_document)
        elif os.path.exists(source):
            source_path = Path(source)
            self._source = source
            logger.info(f"Reading data from folder: {source}")
            file_list = os.listdir(source)

            # Separate files by type for limiting
            text_files = [f for f in file_list if f.endswith(".txt")]
            pdf_files = [f for f in file_list if f.endswith(".pdf")]
            csv_files = [f for f in file_list if f.endswith(".csv")]

            # Apply limits to text and PDF files
            text_pdf_count = 0
            text_pdf_limit = (
                max_text_files if max_text_files is not None else float("inf")
            )

            if max_text_files is not None:
                logger.info(
                    f"Limiting import to maximum {max_text_files} text/PDF files"
                )

            # Process text files
            for file_name in tqdm(
                text_files, desc="Reading text files", disable=len(text_files) < 10
            ):
                if text_pdf_count >= text_pdf_limit:
                    break

                file_path = source_path / file_name
                with open(file_path) as f:
                    read_from_file = f.read()
                    read_from_file = apply_ignore_patterns(read_from_file)
                    # Extract timestamp from content
                    doc_timestamp = extract_timestamp_from_text(read_from_file)
                    # Extract tag from filename if it contains - or _
                    tag = extract_tag_from_filename(file_name)
                    # self._content removed
                    metadata = {
                        "source": str(file_path),
                        "file_name": file_name,
                    }
                    if tag:
                        metadata["tag"] = tag
                    _document = Document(
                        text=read_from_file,
                        timestamp=doc_timestamp,
                        metadata=metadata,
                        id=file_name,
                        score=0.0,
                        name="",
                        description="",
                    )
                    self._documents.append(_document)
                    text_pdf_count += 1

            # Process PDF files
            for file_name in tqdm(
                pdf_files, desc="Reading PDF files", disable=len(pdf_files) < 10
            ):
                if text_pdf_count >= text_pdf_limit:
                    break

                file_path = source_path / file_name
                with open(file_path, "rb") as f:
                    reader = PdfReader(f)
                    # Use list and join for efficient string concatenation
                    page_texts = []
                    for page in tqdm(
                        reader.pages,
                        desc=f"Reading PDF {file_name}",
                        leave=False,
                        disable=len(reader.pages) < 10,
                    ):
                        page_texts.append(page.extract_text())
                    read_from_file = "".join(page_texts)
                    read_from_file = apply_ignore_patterns(read_from_file)
                    # Extract timestamp from content
                    doc_timestamp = extract_timestamp_from_text(read_from_file)
                    # Extract tag from filename if it contains - or _
                    tag = extract_tag_from_filename(file_name)
                    # self._content removed
                    metadata = {
                        "source": str(file_path),
                        "file_name": file_name,
                    }
                    if tag:
                        metadata["tag"] = tag
                    _document = Document(
                        text=read_from_file,
                        timestamp=doc_timestamp,
                        metadata=metadata,
                        id=file_name,
                        score=0.0,
                        name="",
                        description="",
                    )
                    self._documents.append(_document)
                    text_pdf_count += 1

            # Process CSV files
            for file_name in csv_files:
                file_path = source_path / file_name
                if comma_separated_text_columns == "":
                    logger.info(f"Reading CSV file: {file_path}")
                    self._df = Csv().read_csv(file_path)
                    # Apply row limit if specified
                    if max_csv_rows is not None and len(self._df) > max_csv_rows:
                        logger.info(f"Limiting CSV to first {max_csv_rows} rows")
                        self._df = self._df.head(max_csv_rows)
                    logger.info(f"CSV file read with shape: {self._df.shape}")
                    _CSV_EXISTS = True
                if comma_separated_text_columns != "":
                    logger.info(f"Reading CSV file to corpus: {file_path}")
                    self.read_csv_to_corpus(
                        file_path,
                        comma_separated_ignore_words,
                        comma_separated_text_columns,
                        max_rows=max_csv_rows,
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
        Convert the corpus to a pandas dataframe. Parallelizes for large corpora.
        """
        if not self._corpus:
            raise ValueError("No corpus found. Please create a corpus first.")
        documents = self._corpus.documents
        if len(documents) < 10:
            data = [
                document.model_dump()
                for document in tqdm(
                    documents, desc="Converting to dataframe", disable=True
                )
            ]
        else:
            data = []

            def dump_doc(document):
                return document.model_dump()

            n_cores = multiprocessing.cpu_count()
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(dump_doc, document): document
                    for document in documents
                }
                with tqdm(
                    total=len(futures),
                    desc=f"Converting to dataframe (parallel, {n_cores} cores)",
                ) as pbar:
                    for future in as_completed(futures):
                        data.append(future.result())
                        pbar.update(1)
        df = pd.DataFrame(data)
        return df
