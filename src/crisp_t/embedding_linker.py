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
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler

from .model import Corpus

logger = logging.getLogger(__name__)

CHROMADB_AVAILABLE = False
try:
    import chromadb  # noqa: F401

    CHROMADB_AVAILABLE = True
except ImportError:
    logger.warning("ChromaDB not available. Install with: pip install chromadb")


class EmbeddingLinker:
    """
    Embedding-based cross-modal linking between text documents and numeric data.

    This class provides fuzzy semantic alignment when explicit IDs or timestamps
    are missing, complementing existing ID-based, keyword-based, and time-based
    linking methods in CRISP-T.

    Methods:
    - Text embeddings: Uses sentence transformers or ChromaDB's default embeddings
    - Numeric embeddings: Standardized numeric vectors from dataframe rows
    - Similarity: Cosine similarity or Euclidean distance in embedding space
    - Linking: Nearest neighbor search with optional threshold filtering
    """

    def __init__(
        self,
        corpus: Corpus,
        text_embedding_model: str = "all-MiniLM-L6-v2",
        similarity_metric: str = "cosine",
        use_simple_embeddings: bool = False,
    ):
        """
        Initialize the EmbeddingLinker.

        Args:
            corpus: Corpus object with documents and dataframe
            text_embedding_model: Name of sentence transformer model or 'simple'
            similarity_metric: 'cosine' or 'euclidean'
            use_simple_embeddings: Use simple TF-IDF based embeddings (no downloads)
        """
        self.corpus = corpus
        self.text_embedding_model = text_embedding_model
        self.similarity_metric = similarity_metric
        self.use_simple_embeddings = use_simple_embeddings

        # Cache for embeddings
        self._text_embeddings = None
        self._numeric_embeddings = None
        self._text_ids = []
        self._numeric_indices = []

    def _get_text_embeddings(self) -> np.ndarray:
        """
        Compute or retrieve cached text embeddings for all documents.

        Uses ChromaDB with sentence transformers for high-quality embeddings,
        or falls back to simple TF-IDF based embeddings.

        Returns:
            Array of shape (n_documents, embedding_dim)
        """
        if self._text_embeddings is not None:
            return self._text_embeddings

        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB is required for embedding-based linking. "
                "Install with: pip install chromadb"
            )

        from .semantic import Semantic

        # Use Semantic class to generate embeddings
        try:
            semantic = Semantic(
                self.corpus, use_simple_embeddings=self.use_simple_embeddings
            )

            # Get collection to access embeddings
            collection = semantic._collection

            # Retrieve all documents with embeddings
            results = collection.get(include=["embeddings"])

            if results["embeddings"] is None or len(results["embeddings"]) == 0:
                raise ValueError(
                    "No embeddings found. Ensure documents are added to collection."
                )

            self._text_embeddings = np.array(results["embeddings"])
            self._text_ids = results["ids"]

            logger.info(
                f"Generated text embeddings: shape {self._text_embeddings.shape}"
            )

            return self._text_embeddings

        except Exception as e:
            logger.exception(f"Error generating text embeddings: {e}")
            raise

    def _get_numeric_embeddings(
        self, columns: list[str] | None = None, normalize: bool = True
    ) -> np.ndarray:
        """
        Compute or retrieve cached numeric embeddings from dataframe.

        Each row is encoded as a standardized vector of its numeric features.

        Args:
            columns: List of column names to use. If None, uses all numeric columns.
            normalize: Whether to standardize features (recommended)

        Returns:
            Array of shape (n_rows, n_features)
        """
        if self._numeric_embeddings is not None:
            return self._numeric_embeddings

        if self.corpus.df is None or self.corpus.df.empty:
            raise ValueError("Corpus has no dataframe for numeric embeddings")

        # Select columns
        if columns is None:
            # Auto-select numeric columns
            df = self.corpus.df.select_dtypes(include=[np.number])
        else:
            # Use specified columns
            df = self.corpus.df[columns]

        if df.empty or df.shape[1] == 0:
            raise ValueError("No numeric columns found for embedding generation")

        # Handle missing values
        df_clean = df.fillna(df.mean())

        # Convert to numpy array
        numeric_data = df_clean.to_numpy()

        # Normalize if requested
        if normalize:
            scaler = StandardScaler()
            numeric_data = scaler.fit_transform(numeric_data)

        # Replace any remaining NaNs with zeros (handles columns that are all NaN)
        if np.isnan(numeric_data).any():
            logger.warning("NaNs found in numeric embeddings, replacing with zeros.")
            numeric_data = np.nan_to_num(numeric_data, nan=0.0)

        self._numeric_embeddings = numeric_data
        self._numeric_indices = list(df.index)

        logger.info(
            f"Generated numeric embeddings: shape {self._numeric_embeddings.shape}"
        )

        return self._numeric_embeddings

    def compute_similarity_matrix(
        self,
        numeric_columns: list[str] | None = None,
    ) -> np.ndarray:
        """
        Compute similarity matrix between text and numeric embeddings.
        Projects both embeddings to a common lower dimension using PCA.
        Args:
            numeric_columns: Columns to use for numeric embeddings
        Returns:
            Similarity matrix of shape (n_documents, n_rows)
        """
        text_emb = self._get_text_embeddings()
        numeric_emb = self._get_numeric_embeddings(columns=numeric_columns)

        # Check for NaNs in embeddings
        if np.isnan(text_emb).any():
            raise ValueError(
                "Text embeddings contain NaN values. Cannot compute similarity."
            )
        if np.isnan(numeric_emb).any():
            raise ValueError(
                "Numeric embeddings contain NaN values. Cannot compute similarity."
            )

        # Project both to the same lower dimension using PCA
        from sklearn.decomposition import PCA

        # Determine the maximum allowed n_components for PCA
        n_text_samples, n_text_features = text_emb.shape
        n_num_samples, n_num_features = numeric_emb.shape
        max_pca_dim = min(
            n_text_samples, n_num_samples, n_text_features, n_num_features
        )
        if max_pca_dim < 1:
            raise ValueError("Cannot perform PCA: insufficient samples or features.")

        # Only project if dimensions differ
        if text_emb.shape[1] != max_pca_dim:
            pca_text = PCA(n_components=max_pca_dim, random_state=42)
            text_emb_proj = pca_text.fit_transform(text_emb)
        else:
            text_emb_proj = text_emb
        if numeric_emb.shape[1] != max_pca_dim:
            pca_num = PCA(n_components=max_pca_dim, random_state=42)
            numeric_emb_proj = pca_num.fit_transform(numeric_emb)
        else:
            numeric_emb_proj = numeric_emb

        # Compute similarity
        if self.similarity_metric == "cosine":
            similarity = cosine_similarity(text_emb_proj, numeric_emb_proj)
        elif self.similarity_metric == "euclidean":
            distances = euclidean_distances(text_emb_proj, numeric_emb_proj)
            similarity = 1.0 / (1.0 + distances)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

        return similarity

    def link_by_embedding_similarity(
        self,
        numeric_columns: list[str] | None = None,
        threshold: float | None = None,
        top_k: int = 1,
    ) -> Corpus:
        """
        Link documents to dataframe rows by embedding similarity.

        Args:
            numeric_columns: Columns to use for numeric embeddings
            threshold: Minimum similarity threshold (0-1). If None, no filtering.
            top_k: Number of top similar rows to link per document

        Returns:
            Updated corpus with embedding links in document metadata
        """
        logger.info("Computing embedding-based links...")

        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(
            numeric_columns=numeric_columns
        )

        # For each document, find top-k most similar rows
        for i, doc in enumerate(self.corpus.documents):
            if i >= similarity_matrix.shape[0]:
                logger.warning(f"Document index {i} out of range for similarity matrix")
                continue

            similarities = similarity_matrix[i]

            # Get top-k indices
            if top_k == 1:
                top_indices = [np.argmax(similarities)]
            else:
                top_indices = np.argsort(similarities)[-top_k:][::-1]

            # Filter by threshold if specified
            links = []
            for idx in top_indices:
                sim_score = similarities[idx]

                if threshold is None or sim_score >= threshold:
                    df_index = self._numeric_indices[idx]
                    links.append(
                        {
                            "df_index": int(df_index),
                            "similarity_score": float(sim_score),
                            "link_type": "embedding",
                            "similarity_metric": self.similarity_metric,
                        }
                    )

            # Store links in document metadata
            if links:
                if "embedding_links" not in doc.metadata:
                    doc.metadata["embedding_links"] = []
                doc.metadata["embedding_links"].extend(links)
                # write back to corpus
                self.corpus.documents[i] = doc

        linked_count = sum(
            1
            for doc in self.corpus.documents
            if doc.metadata.get("embedding_links")
        )

        logger.info(f"Linked {linked_count} documents using embedding similarity")

        return self.corpus

    def get_link_statistics(self) -> dict[str, Any]:
        """
        Get statistics about embedding-based links.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_documents": len(self.corpus.documents),
            "linked_documents": 0,
            "total_links": 0,
            "avg_similarity": 0.0,
            "min_similarity": 1.0,
            "max_similarity": 0.0,
        }

        similarities = []

        for doc in self.corpus.documents:
            if doc.metadata.get("embedding_links"):
                stats["linked_documents"] += 1
                links = doc.metadata["embedding_links"]
                stats["total_links"] += len(links)

                for link in links:
                    sim = link.get("similarity_score", 0.0)
                    similarities.append(sim)

        if similarities:
            stats["avg_similarity"] = float(np.mean(similarities))
            stats["min_similarity"] = float(np.min(similarities))
            stats["max_similarity"] = float(np.max(similarities))

        return stats

    def visualize_embedding_space(
        self,
        output_path: str | None = None,
        method: str = "tsne",
    ):
        """
        Visualize text and numeric embeddings in 2D space using dimensionality reduction.

        Args:
            output_path: Path to save the plot
            method: 'tsne', 'pca', or 'umap'

        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
        except ImportError:
            raise ImportError("matplotlib and sklearn required for visualization") from None

        text_emb = self._get_text_embeddings()
        numeric_emb = self._get_numeric_embeddings()

        # Combine embeddings
        all_emb = np.vstack([text_emb, numeric_emb])

        # Reduce to 2D
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
        elif method == "pca":
            reducer = PCA(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown reduction method: {method}")

        coords_2d = reducer.fit_transform(all_emb)

        # Split back into text and numeric
        text_coords = coords_2d[: len(text_emb)]
        numeric_coords = coords_2d[len(text_emb) :]

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(
            text_coords[:, 0],
            text_coords[:, 1],
            c="blue",
            label="Text Documents",
            alpha=0.6,
            s=50,
        )

        ax.scatter(
            numeric_coords[:, 0],
            numeric_coords[:, 1],
            c="red",
            label="Numeric Rows",
            alpha=0.6,
            s=50,
            marker="s",
        )

        ax.set_xlabel(f"{method.upper()} Component 1")
        ax.set_ylabel(f"{method.upper()} Component 2")
        ax.set_title("Text-Numeric Embedding Space")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved visualization to {output_path}")

        return fig
