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
from typing import Optional, List, Set, Tuple
import pandas as pd
from .model import Corpus, Document

logger = logging.getLogger(__name__)


class LinkageFilter:
    """
    Advanced filtering that supports multiple linkage methods.
    
    Extends basic ID-based filtering to support:
    - ID-based linkage (existing)
    - Keyword-based linkage
    - Time-based linkage (temporal_links)
    - Embedding-based linkage (embedding_links)
    
    Uses existing linkage metadata to filter text and numeric data.
    """
    
    LINKAGE_METHODS = ["id", "keyword", "time", "embedding"]
    
    def __init__(self, corpus: Corpus):
        """
        Initialize the LinkageFilter.
        
        Args:
            corpus: Corpus with documents and dataframe
        """
        self.corpus = corpus
        self.original_corpus = None  # Store original for reset
    
    def filter_by_linkage(
        self,
        method: str,
        key: str,
        value: str,
        min_similarity: Optional[float] = None,
    ) -> Corpus:
        """
        Filter corpus using specified linkage method.
        
        Args:
            method: Linkage method ('id', 'keyword', 'time', 'embedding')
            key: Filter key (column name for id/keyword, 'timestamp' for time)
            value: Filter value to match
            min_similarity: Minimum similarity for embedding links (0-1)
            
        Returns:
            Filtered corpus
            
        Raises:
            ValueError: If linkage metadata is missing or method is invalid
        """
        if method not in self.LINKAGE_METHODS:
            raise ValueError(
                f"Invalid linkage method '{method}'. "
                f"Must be one of: {', '.join(self.LINKAGE_METHODS)}"
            )
        
        # Dispatch to specific filter method
        if method == "id":
            return self._filter_by_id(key, value)
        elif method == "keyword":
            return self._filter_by_keyword(key, value)
        elif method == "time":
            return self._filter_by_time(key, value)
        elif method == "embedding":
            return self._filter_by_embedding(key, value, min_similarity)
        
        return self.corpus
    
    def _filter_by_id(self, column: str, value: str) -> Corpus:
        """
        Filter by ID-based linkage (existing functionality).
        
        Filters dataframe rows where column == value, then filters
        documents by matching IDs.
        """
        if self.corpus.df is None or self.corpus.df.empty:
            raise ValueError("No dataframe available for ID-based filtering")
        
        if column not in self.corpus.df.columns:
            raise ValueError(
                f"Column '{column}' not found in dataframe. "
                f"Available columns: {', '.join(self.corpus.df.columns)}"
            )
        
        # Filter dataframe
        filtered_df = self.corpus.df[self.corpus.df[column] == value]
        
        # Try int conversion if no results
        if filtered_df.empty:
            try:
                filtered_df = self.corpus.df[self.corpus.df[column] == int(value)]
            except (ValueError, TypeError):
                pass
        
        if filtered_df.empty:
            logger.warning(f"No rows match {column} == {value}")
            return self.corpus
        
        # Get valid IDs (assuming 'id' column exists)
        id_col = "id"
        for potential_id in ["id", "ID", "doc_id", "document_id"]:
            if potential_id in filtered_df.columns:
                id_col = potential_id
                break
        
        if id_col not in filtered_df.columns:
            logger.warning("No ID column found for document filtering")
            # Return corpus with filtered dataframe only
            new_corpus = Corpus(
                id=self.corpus.id,
                name=self.corpus.name,
                description=self.corpus.description,
                documents=self.corpus.documents,
                df=filtered_df,
                metadata=self.corpus.metadata.copy(),
            )
            return new_corpus
        
        # Filter documents by ID
        valid_ids = set(filtered_df[id_col].tolist())
        filtered_docs = [
            doc for doc in self.corpus.documents
            if hasattr(doc, id_col) and getattr(doc, id_col, None) in valid_ids
        ]
        
        logger.info(
            f"ID filter: {len(filtered_df)} rows, {len(filtered_docs)} documents"
        )
        
        # Create new corpus
        new_corpus = Corpus(
            id=self.corpus.id,
            name=self.corpus.name,
            description=self.corpus.description,
            documents=filtered_docs,
            df=filtered_df,
            metadata=self.corpus.metadata.copy(),
        )
        
        return new_corpus
    
    def _filter_by_keyword(self, keyword: str, value: str) -> Corpus:
        """
        Filter by keyword-based linkage.
        
        Filters documents that have the keyword in metadata,
        then filters dataframe rows linked via relationships.
        """
        # Filter documents by keyword
        filtered_docs = []
        for doc in self.corpus.documents:
            if "keywords" in doc.metadata:
                keywords = doc.metadata["keywords"]
                if isinstance(keywords, str):
                    keywords = [kw.strip() for kw in keywords.split(",")]
                if keyword in keywords:
                    filtered_docs.append(doc)
        
        if not filtered_docs:
            raise ValueError(
                f"No documents found with keyword '{keyword}'. "
                "Hint: Run topic modeling and keyword assignment first with: "
                "crisp --inp <folder> --topics --assign --out <folder>"
            )
        
        # Get relationships involving this keyword
        relationships = self.corpus.get_relationships()
        linked_columns = set()
        
        for rel in relationships:
            if f"text:{keyword}" in rel.get("first", "") or keyword in rel.get("first", ""):
                # Extract column name from second (numb:column_name)
                second = rel.get("second", "")
                if second.startswith("numb:"):
                    col = second.replace("numb:", "")
                    linked_columns.add(col)
        
        if not linked_columns:
            logger.warning(
                f"No relationships found for keyword '{keyword}'. "
                "Returning documents without dataframe filtering."
            )
            # Return corpus with filtered documents only
            new_corpus = Corpus(
                id=self.corpus.id,
                name=self.corpus.name,
                description=self.corpus.description,
                documents=filtered_docs,
                df=self.corpus.df,
                metadata=self.corpus.metadata.copy(),
            )
            return new_corpus
        
        # Filter dataframe if value provided
        filtered_df = self.corpus.df
        if value and self.corpus.df is not None:
            # Filter by first linked column
            first_col = list(linked_columns)[0]
            if first_col in self.corpus.df.columns:
                filtered_df = self.corpus.df[self.corpus.df[first_col] == value]
        
        logger.info(
            f"Keyword filter: {len(filtered_docs)} documents, "
            f"{len(filtered_df) if filtered_df is not None else 0} rows"
        )
        
        new_corpus = Corpus(
            id=self.corpus.id,
            name=self.corpus.name,
            description=self.corpus.description,
            documents=filtered_docs,
            df=filtered_df,
            metadata=self.corpus.metadata.copy(),
        )
        
        return new_corpus
    
    def _filter_by_time(self, time_column: str, value: str) -> Corpus:
        """
        Filter by time-based linkage.
        
        Uses temporal_links metadata to filter documents and dataframe rows
        that are linked within the specified time range or value.
        """
        # Check if documents have temporal links
        has_temporal_links = any(
            "temporal_links" in doc.metadata and doc.metadata["temporal_links"]
            for doc in self.corpus.documents
        )
        
        if not has_temporal_links:
            raise ValueError(
                "No temporal links found in corpus. "
                "Hint: Create temporal links first:\n"
                "  crispt --inp <folder> --temporal-link 'nearest:timestamp' --out <folder>\n"
                "  or use window/sequence methods. See notes/TEMPORAL_ANALYSIS.md"
            )
        
        # Filter documents that have temporal links
        filtered_docs = []
        linked_row_indices = set()
        
        for doc in self.corpus.documents:
            if "temporal_links" in doc.metadata and doc.metadata["temporal_links"]:
                filtered_docs.append(doc)
                # Collect linked row indices
                for link in doc.metadata["temporal_links"]:
                    linked_row_indices.add(link.get("df_index"))
        
        # Filter dataframe by linked indices
        filtered_df = None
        if self.corpus.df is not None and linked_row_indices:
            # Convert to list and filter
            valid_indices = [idx for idx in linked_row_indices if idx in self.corpus.df.index]
            filtered_df = self.corpus.df.loc[valid_indices]
        
        logger.info(
            f"Time filter: {len(filtered_docs)} documents, "
            f"{len(filtered_df) if filtered_df is not None else 0} rows"
        )
        
        new_corpus = Corpus(
            id=self.corpus.id,
            name=self.corpus.name,
            description=self.corpus.description,
            documents=filtered_docs,
            df=filtered_df,
            metadata=self.corpus.metadata.copy(),
        )
        
        return new_corpus
    
    def _filter_by_embedding(
        self,
        key: str,
        value: str,
        min_similarity: Optional[float] = None
    ) -> Corpus:
        """
        Filter by embedding-based linkage.
        
        Uses embedding_links metadata to filter documents and dataframe rows
        that are linked with sufficient similarity.
        """
        # Check if documents have embedding links
        has_embedding_links = any(
            "embedding_links" in doc.metadata and doc.metadata["embedding_links"]
            for doc in self.corpus.documents
        )
        
        if not has_embedding_links:
            raise ValueError(
                "No embedding links found in corpus. "
                "Hint: Create embedding links first:\n"
                "  crispt --inp <folder> --embedding-link 'cosine:1:0.7' --out <folder>\n"
                "  See notes/EMBEDDING_LINKING.md for details"
            )
        
        # Filter documents with embedding links above threshold
        filtered_docs = []
        linked_row_indices = set()
        
        threshold = min_similarity if min_similarity is not None else 0.0
        
        for doc in self.corpus.documents:
            if "embedding_links" in doc.metadata and doc.metadata["embedding_links"]:
                # Check if any link meets similarity threshold
                valid_links = [
                    link for link in doc.metadata["embedding_links"]
                    if link.get("similarity_score", 0) >= threshold
                ]
                
                if valid_links:
                    filtered_docs.append(doc)
                    # Collect linked row indices
                    for link in valid_links:
                        linked_row_indices.add(link.get("df_index"))
        
        if not filtered_docs:
            raise ValueError(
                f"No embedding links with similarity >= {threshold}. "
                f"Try lowering the threshold or re-link with different parameters."
            )
        
        # Filter dataframe by linked indices
        filtered_df = None
        if self.corpus.df is not None and linked_row_indices:
            valid_indices = [idx for idx in linked_row_indices if idx in self.corpus.df.index]
            filtered_df = self.corpus.df.loc[valid_indices]
        
        logger.info(
            f"Embedding filter (similarity >= {threshold}): "
            f"{len(filtered_docs)} documents, "
            f"{len(filtered_df) if filtered_df is not None else 0} rows"
        )
        
        new_corpus = Corpus(
            id=self.corpus.id,
            name=self.corpus.name,
            description=self.corpus.description,
            documents=filtered_docs,
            df=filtered_df,
            metadata=self.corpus.metadata.copy(),
        )
        
        return new_corpus
    
    def get_filter_statistics(self, method: str) -> dict:
        """
        Get statistics about available filters for a linkage method.
        
        Args:
            method: Linkage method to get stats for
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            "method": method,
            "total_documents": len(self.corpus.documents),
            "linkage_available": False,
            "linked_documents": 0,
        }
        
        if method == "time":
            linked = sum(
                1 for doc in self.corpus.documents
                if "temporal_links" in doc.metadata and doc.metadata["temporal_links"]
            )
            stats["linkage_available"] = linked > 0
            stats["linked_documents"] = linked
            
        elif method == "embedding":
            linked = sum(
                1 for doc in self.corpus.documents
                if "embedding_links" in doc.metadata and doc.metadata["embedding_links"]
            )
            stats["linkage_available"] = linked > 0
            stats["linked_documents"] = linked
            
        elif method == "keyword":
            linked = sum(
                1 for doc in self.corpus.documents
                if "keywords" in doc.metadata
            )
            stats["linkage_available"] = linked > 0
            stats["linked_documents"] = linked
            
        return stats
