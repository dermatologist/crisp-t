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

import os
import warnings
from typing import Optional

import pandas as pd

from .model import Corpus, Document

warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.api.types import EmbeddingFunction

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    EmbeddingFunction = object  # type: ignore


class SimpleEmbeddingFunction(EmbeddingFunction):
    """
    A simple embedding function for testing that doesn't require downloads.
    Uses TF-IDF based embeddings with a fixed vocabulary.
    """

    def __init__(self):
        """Initialize with an empty vocabulary that will be built from data."""
        self._vocabulary = set()
        self._word_to_idx = {}

    def _build_vocabulary(self, texts: list[str]):
        """Build vocabulary from texts."""
        for text in texts:
            words = text.lower().split()
            self._vocabulary.update(words)
        # Sort words for consistent ordering
        word_list = sorted(self._vocabulary)
        self._word_to_idx = {word: idx for idx, word in enumerate(word_list)}

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Generate simple embeddings based on word presence."""
        # Build vocabulary if not already built
        if not self._word_to_idx:
            self._build_vocabulary(input)

        # Create embeddings
        embeddings = []
        for text in input:
            words = text.lower().split()
            # Create a vector of size len(vocabulary)
            embedding = [0.0] * len(self._word_to_idx)
            for word in words:
                if word in self._word_to_idx:
                    embedding[self._word_to_idx[word]] += 1.0
            # Normalize
            total = sum(embedding)
            if total > 0:
                embedding = [x / total for x in embedding]
            # Ensure we have at least some value
            if total == 0:
                embedding = [1.0 / len(embedding)] * len(embedding)
            embeddings.append(embedding)

        return embeddings


class Semantic:
    """
    Semantic search class using ChromaDB for similarity-based document retrieval.
    """

    def __init__(self, corpus: Corpus, use_simple_embeddings: bool = False):
        """
        Initialize the Semantic class with a corpus.

        Args:
            corpus: The Corpus object containing documents to index.
            use_simple_embeddings: If True, use simple embeddings instead of default (useful for testing).

        Raises:
            ImportError: If chromadb is not installed.
            ValueError: If corpus is None or has no documents.
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb is required for semantic search. "
                "Please install it with: pip install chromadb"
            )

        if corpus is None:
            raise ValueError("Corpus cannot be None")

        if not corpus.documents:
            raise ValueError("Corpus must contain at least one document")

        self._corpus = corpus
        self._client = chromadb.Client(Settings(anonymized_telemetry=False))
        self._collection_name = "crisp-t"
        self._embedding_function = None

        # Pre-build vocabulary for simple embeddings
        if use_simple_embeddings:
            self._embedding_function = SimpleEmbeddingFunction()
            # Build vocabulary from all document texts
            all_texts = [doc.text for doc in corpus.documents]
            self._embedding_function._build_vocabulary(all_texts)

        # Create or get collection - delete existing if using different embedding
        try:
            existing_collection = self._client.get_collection(name=self._collection_name)
            # Delete and recreate to ensure clean state
            self._client.delete_collection(name=self._collection_name)
        except Exception:
            pass  # Collection doesn't exist yet

        # Create new collection
        if use_simple_embeddings and self._embedding_function:
            # Use simple embeddings for testing
            self._collection = self._client.create_collection(
                name=self._collection_name,
                embedding_function=self._embedding_function,
            )
        else:
            # Use default embeddings (may require download)
            self._collection = self._client.create_collection(name=self._collection_name)

        # Add documents to collection
        self._add_documents_to_collection()

    def _add_documents_to_collection(self):
        """
        Add corpus documents to the ChromaDB collection.
        """
        documents_texts = []
        metadatas = []
        ids = []

        for doc in self._corpus.documents:
            documents_texts.append(doc.text)
            # Prepare metadata - ChromaDB requires string values
            metadata = {}
            for key, value in doc.metadata.items():
                # Convert non-string values to strings
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = str(value)
                elif isinstance(value, (list, tuple)):
                    metadata[key] = str(value)
                else:
                    metadata[key] = str(value)
            # Add document name if available
            if doc.name:
                metadata["name"] = doc.name
            metadatas.append(metadata)
            ids.append(str(doc.id))

        # Add to collection
        self._collection.add(documents=documents_texts, metadatas=metadatas, ids=ids)

    def get_similar(self, query: str, n_results: int = 5) -> Corpus:
        """
        Perform semantic search and return similar documents as a new Corpus.

        Args:
            query: The search query string.
            n_results: Number of similar documents to return (default: 5).

        Returns:
            A new Corpus containing the most similar documents.
        """
        # Query the collection
        results = self._collection.query(query_texts=[query], n_results=n_results)

        # Create a new corpus with the results
        similar_documents = []
        result_ids = results["ids"][0] if results["ids"] else []

        for doc_id in result_ids:
            # Find the document in the original corpus
            doc = self._corpus.get_document_by_id(doc_id)
            if doc:
                similar_documents.append(doc)

        # Create new corpus with similar documents
        new_corpus = Corpus(
            id=f"{self._corpus.id}_semantic_search",
            name=f"{self._corpus.name or 'Corpus'} - Semantic Search Results",
            description=f"Semantic search results for query: {query}",
            documents=similar_documents,
            df=self._corpus.df,
            visualization=self._corpus.visualization.copy(),
            metadata=self._corpus.metadata.copy(),
        )

        # Update metadata with search query
        new_corpus.metadata["semantic_query"] = query
        new_corpus.metadata["semantic_n_results"] = n_results

        # Update self.corpus for consistency
        self._corpus = new_corpus

        return new_corpus

    def get_df(self, metadata_keys: Optional[list[str]] = None) -> Corpus:
        """
        Export collection metadata as a pandas DataFrame and merge with corpus.df.

        Args:
            metadata_keys: List of metadata keys to include. If None, include all.

        Returns:
            Updated Corpus with metadata integrated into the DataFrame.
        """
        # Get all documents from the collection
        all_results = self._collection.get()

        # Extract ids and metadatas
        ids = all_results["ids"]
        metadatas = all_results["metadatas"]

        # Create a list of dictionaries for the DataFrame
        records = []
        for doc_id, metadata in zip(ids, metadatas):
            record = {"id": doc_id}
            if metadata_keys:
                # Only include specified keys
                for key in metadata_keys:
                    if key in metadata:
                        record[key] = metadata[key]
            else:
                # Include all metadata
                record.update(metadata)
            records.append(record)

        # Create DataFrame from records
        metadata_df = pd.DataFrame(records)

        # Try to merge with existing dataframe
        if self._corpus.df is not None and not self._corpus.df.empty:
            # Try to merge on 'id' column if it exists in corpus.df
            if "id" in self._corpus.df.columns:
                try:
                    # Merge the dataframes
                    merged_df = pd.merge(
                        self._corpus.df,
                        metadata_df,
                        on="id",
                        how="outer",
                        suffixes=("", "_metadata"),
                    )
                    self._corpus.df = merged_df
                except Exception as e:
                    print(
                        f"WARNING: Could not merge with existing DataFrame: {e}. "
                        "Creating new DataFrame with metadata only."
                    )
                    self._corpus.df = metadata_df
            else:
                print(
                    "WARNING: Existing DataFrame does not have 'id' column. "
                    "Creating new DataFrame with metadata only."
                )
                self._corpus.df = metadata_df
        else:
            # No existing dataframe, use metadata_df
            self._corpus.df = metadata_df

        return self._corpus

    def save_collection(self, path: Optional[str] = None):
        """
        Save the ChromaDB collection to disk.

        Args:
            path: Directory path to save the collection. If None, uses default location.
        """
        if path is None:
            path = "./chromadb_storage"

        # Create persistent client and copy data
        persistent_client = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))

        # Get all data from in-memory collection
        all_data = self._collection.get()

        # Create or get collection in persistent storage
        try:
            persistent_collection = persistent_client.get_collection(
                name=self._collection_name
            )
            # Delete and recreate to ensure fresh data
            persistent_client.delete_collection(name=self._collection_name)
            persistent_collection = persistent_client.create_collection(
                name=self._collection_name
            )
        except Exception:
            persistent_collection = persistent_client.create_collection(
                name=self._collection_name
            )

        # Add data to persistent collection
        if all_data["ids"]:
            persistent_collection.add(
                ids=all_data["ids"],
                documents=all_data["documents"],
                metadatas=all_data["metadatas"],
            )

        print(f"Collection saved to {path}")

    def restore_collection(self, path: Optional[str] = None):
        """
        Restore the ChromaDB collection from disk.

        Args:
            path: Directory path to restore the collection from. If None, uses default location.
        """
        if path is None:
            path = "./chromadb_storage"

        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")

        # Create persistent client
        persistent_client = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))

        # Get collection from persistent storage
        persistent_collection = persistent_client.get_collection(name=self._collection_name)

        # Get all data
        all_data = persistent_collection.get()

        # Clear current in-memory collection
        try:
            self._client.delete_collection(name=self._collection_name)
        except Exception:
            pass

        # Create new in-memory collection
        self._collection = self._client.create_collection(name=self._collection_name)

        # Add data to in-memory collection
        if all_data["ids"]:
            self._collection.add(
                ids=all_data["ids"],
                documents=all_data["documents"],
                metadatas=all_data["metadatas"],
            )

        print(f"Collection restored from {path}")
