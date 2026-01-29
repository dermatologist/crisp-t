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

import logging
import os
import warnings

import pandas as pd
from tqdm import tqdm

from .model import Corpus

warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    import chromadb
    from chromadb.api.types import EmbeddingFunction
    from chromadb.config import Settings

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

    def __init__(self, corpus: Corpus, use_simple_embeddings: bool = False, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the Semantic class with a corpus.

        Args:
            corpus: The Corpus object containing documents to index.
            use_simple_embeddings: If True, use simple embeddings instead of default (useful for testing).
            chunk_size: Size of text chunks in characters for chunk-based search (default: 500).
            chunk_overlap: Overlap between chunks in characters (default: 50).

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
        self._chunks_collection_name = "crisp-t-chunks"
        self._embedding_function = None
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        # Pre-build vocabulary for simple embeddings
        if use_simple_embeddings:
            self._embedding_function = SimpleEmbeddingFunction()
            # Build vocabulary from all document texts
            all_texts = [doc.text for doc in corpus.documents]
            self._embedding_function._build_vocabulary(all_texts)

        # Create or get collection - delete existing if using different embedding
        try:
            self._client.get_collection(name=self._collection_name)
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

        # Create chunks collection - delete existing if present
        try:
            self._client.get_collection(name=self._chunks_collection_name)
            self._client.delete_collection(name=self._chunks_collection_name)
        except Exception:
            pass  # Collection doesn't exist yet

        # Create chunks collection
        if use_simple_embeddings and self._embedding_function:
            self._chunks_collection = self._client.create_collection(
                name=self._chunks_collection_name,
                embedding_function=self._embedding_function,
            )
        else:
            self._chunks_collection = self._client.create_collection(name=self._chunks_collection_name)

        # Add document chunks to collection
        self._add_chunks_to_collection()

    def _add_documents_to_collection(self):
        """
        Add corpus documents to the ChromaDB collection.
        """
        documents_texts = []
        metadatas = []
        ids = []

        for doc in tqdm(self._corpus.documents, desc="Adding documents to collection", disable=len(self._corpus.documents) < 10):
            documents_texts.append(doc.text)
            # Prepare metadata - ChromaDB requires string values and non-empty dicts
            metadata = {}
            for key, value in doc.metadata.items():
                # Convert non-string values to strings
                if isinstance(value, (str, int, float, bool)) or isinstance(value, (list, tuple)):
                    metadata[key] = str(value)
                else:
                    metadata[key] = str(value)
            # Add document name if available
            if doc.name:
                metadata["name"] = doc.name
            # ChromaDB requires non-empty metadata, add document id if empty
            if not metadata:
                metadata["_doc_id"] = str(doc.id)
            metadatas.append(metadata)
            ids.append(str(doc.id))

        # Add to collection
        self._collection.add(documents=documents_texts, metadatas=metadatas, ids=ids)

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: The text to chunk.

        Returns:
            List of text chunks.
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Get chunk from start to start + chunk_size
            end = min(start + self._chunk_size, text_length)
            chunk = text[start:end]

            # Only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk)

            # Move start position by chunk_size - overlap
            start += self._chunk_size - self._chunk_overlap

            # Break if we've reached the end
            if end >= text_length:
                break

        return chunks

    def _add_chunks_to_collection(self):
        """
        Chunk documents and add to the chunks collection.
        """
        chunk_texts = []
        chunk_metadatas = []
        chunk_ids = []
        chunk_counter = 0

        for doc in tqdm(self._corpus.documents, desc="Adding chunks to collection", disable=len(self._corpus.documents) < 10):
            # Chunk the document text
            chunks = self._chunk_text(doc.text)

            for idx, chunk in enumerate(chunks):
                chunk_texts.append(chunk)

                # Create metadata for the chunk
                metadata = {
                    "doc_id": str(doc.id),
                    "chunk_index": str(idx),
                    "total_chunks": str(len(chunks)),
                }

                # Add document name if available
                if doc.name:
                    metadata["doc_name"] = doc.name

                chunk_metadatas.append(metadata)
                chunk_ids.append(f"{doc.id}_chunk_{chunk_counter}")
                chunk_counter += 1

        # Add chunks to collection
        if chunk_texts:
            self._chunks_collection.add(
                documents=chunk_texts, metadatas=chunk_metadatas, ids=chunk_ids
            )

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

    def get_similar_documents(
        self, document_ids: str, n_results: int = 5, threshold: float = 0.7
    ) -> list[str]:
        """
        Find documents similar to a given set of documents based on semantic similarity.

        This method is useful for literature reviews to find documents that are similar
        to a set of reference documents.

        Args:
            document_ids: A single document ID or comma-separated list of document IDs.
            n_results: Number of similar documents to return (default: 5).
            threshold: Minimum similarity threshold (0-1). Only documents with similarity
                      above this value will be returned (default: 0.7).

        Returns:
            List of document IDs that are similar to the input documents.
        """
        # Parse document IDs (handle single or comma-separated list)
        if isinstance(document_ids, str):
            doc_id_list = [doc_id.strip() for doc_id in document_ids.split(",")]
        else:
            doc_id_list = [document_ids]

        # Get the text content of the reference documents
        reference_texts = []
        for doc_id in doc_id_list:
            doc = self._corpus.get_document_by_id(doc_id)
            if doc:
                reference_texts.append(doc.text)
            else:
                logger.warning(f"Document ID '{doc_id}' not found in corpus")

        if not reference_texts:
            logger.warning("No valid reference documents found")
            return []

        # Combine reference texts into a single query
        # We use all reference texts to find similar documents
        combined_query = " ".join(reference_texts)

        # Query the collection for similar documents
        results = self._collection.query(
            query_texts=[combined_query], n_results=n_results + len(doc_id_list)
        )

        # Extract matching document IDs with their distances
        matching_doc_ids = []
        if results["ids"] and results["distances"]:
            result_ids = results["ids"][0]
            distances = results["distances"][0]

            # Convert distance to similarity (lower distance = higher similarity)
            for doc_id, distance in zip(result_ids, distances):
                # Skip reference documents themselves
                if doc_id in doc_id_list:
                    continue

                # Convert distance to similarity score (0-1 range)
                # For cosine distance: similarity = 1 - (distance / 2)
                similarity = 1 - (distance / 2)
                logger.info(f"Document: {doc_id} | Similarity: {similarity:.4f}")

                # Only include documents above threshold
                if similarity >= threshold:
                    matching_doc_ids.append(doc_id)

                # Stop if we have enough results
                if len(matching_doc_ids) >= n_results:
                    break

        return matching_doc_ids

    def get_similar_chunks(
        self, query: str, doc_id: str, threshold: float = 0.5, n_results: int = 10
    ) -> list[str]:
        """
        Perform semantic search on chunks of a specific document and return matching chunks.

        This method is useful for coding/annotating documents by finding relevant sections
        that match specific concepts or themes.

        Args:
            query: The search query string (concept or set of concepts).
            doc_id: The document ID to search within.
            threshold: Minimum similarity threshold (0-1). Only chunks with similarity
                      above this value will be returned (default: 0.5).
            n_results: Maximum number of chunks to retrieve before filtering (default: 10).

        Returns:
            List of chunk texts that match the query above the threshold.
        """
        # Query the chunks collection
        results = self._chunks_collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"doc_id": str(doc_id)},  # Filter by document ID
        )

        # Extract matching chunks with their distances
        matching_chunks = []
        if results["documents"] and results["distances"]:
            chunks = results["documents"][0]
            distances = results["distances"][0]

            # Convert distance to similarity (lower distance = higher similarity)
            # ChromaDB uses distance metrics, so we need to convert to similarity
            for chunk, distance in zip(chunks, distances):
                # Convert distance to similarity score (0-1 range)
                # For cosine distance: similarity = 1 - (distance / 2)
                similarity = 1 - (distance / 2)
                logger.info(f"Document: {doc_id} | Similarity: {similarity:.4f}")
                # Only include chunks above threshold
                if similarity >= threshold:
                    matching_chunks.append(chunk)

        return matching_chunks

    def get_df(self, metadata_keys: list[str] | None = None) -> Corpus:
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

    def save_collection(self, path: str | None = None):
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

        # Create or get collection in persistent storage with same embedding function
        try:
            persistent_collection = persistent_client.get_collection(
                name=self._collection_name
            )
            # Delete and recreate to ensure fresh data
            persistent_client.delete_collection(name=self._collection_name)
        except Exception:
            pass

        # Create collection with the same embedding function if available
        if self._embedding_function:
            persistent_collection = persistent_client.create_collection(
                name=self._collection_name,
                embedding_function=self._embedding_function
            )
        else:
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

    def restore_collection(self, path: str | None = None):
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

        # Create new in-memory collection with same embedding function
        if self._embedding_function:
            self._collection = self._client.create_collection(
                name=self._collection_name,
                embedding_function=self._embedding_function
            )
        else:
            self._collection = self._client.create_collection(name=self._collection_name)

        # Add data to in-memory collection
        if all_data["ids"]:
            self._collection.add(
                ids=all_data["ids"],
                documents=all_data["documents"],
                metadatas=all_data["metadatas"],
            )

        print(f"Collection restored from {path}")
