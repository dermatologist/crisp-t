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
from typing import Any, Dict, List, Tuple

import networkx as nx
import pandas as pd

from .model.corpus import Corpus

logger = logging.getLogger(__name__)


class CrispGraph:
    def _generate_keyword_metadata_edges(self):
        """
        Generator for edges connecting keywords to metadata nodes.

        Yields:
            Tuple of (source_id, target_id, label, properties) for each edge
        """
        # Collect all keywords
        keywords = set()
        for doc in self.corpus.documents:
            if doc.metadata and "keywords" in doc.metadata:
                doc_keywords = doc.metadata["keywords"]
                if isinstance(doc_keywords, list):
                    keywords.update(doc_keywords)
                elif isinstance(doc_keywords, str):
                    keywords.update([kw.strip() for kw in doc_keywords.split(",")])
        # Collect all metadata node ids
        metadata_ids = set()
        if self.corpus.df is not None and not self.corpus.df.empty:
            df_columns = list(self.corpus.df.columns)
            id_col = None
            for potential_id in ["id", "ID", "doc_id", "document_id", "index"]:
                if potential_id in df_columns:
                    id_col = potential_id
                    break
            if id_col:
                for idx, row in self.corpus.df.iterrows():
                    doc_id = row[id_col]
                    metadata_ids.add(doc_id)
        # Create edges between every keyword and every metadata node
        for keyword in keywords:
            for meta_id in metadata_ids:
                yield (
                    f"keyword:{keyword}",
                    f"metadata:{meta_id}",
                    "KEYWORD_HAS_METADATA",
                    {},
                )

    def _generate_cluster_metadata_edges(self):
        """
        Generator for edges connecting clusters to metadata nodes.

        Yields:
            Tuple of (source_id, target_id, label, properties) for each edge
        """
        # Collect all clusters
        clusters = set()
        for doc in self.corpus.documents:
            if doc.metadata and "cluster" in doc.metadata:
                clusters.add(doc.metadata["cluster"])
        # Collect all metadata node ids
        metadata_ids = set()
        if self.corpus.df is not None and not self.corpus.df.empty:
            df_columns = list(self.corpus.df.columns)
            id_col = None
            for potential_id in ["id", "ID", "doc_id", "document_id", "index"]:
                if potential_id in df_columns:
                    id_col = potential_id
                    break
            if id_col:
                for idx, row in self.corpus.df.iterrows():
                    doc_id = row[id_col]
                    metadata_ids.add(doc_id)
        # Create edges between every cluster and every metadata node
        for cluster_id in clusters:
            for meta_id in metadata_ids:
                yield (
                    f"cluster:{cluster_id}",
                    f"metadata:{meta_id}",
                    "CLUSTER_HAS_METADATA",
                    {},
                )

    """
    Class for creating graph representations of a corpus using NetworkX.

    The graph nodes include:
    - Documents (labelled with IDs)
    - Keywords
    - Clusters (if present)
    - Metadata from DataFrame (if present)

    The graph edges connect keywords, clusters and metadata nodes to the corresponding documents.
    """

    def __init__(self, corpus: Corpus):
        """
        Initialize CrispGraph with a corpus.

        Args:
            corpus: Corpus object to create graph from
        """
        self.corpus = corpus
        self.graph = None

    def _generate_document_nodes(self):
        """
        Generator for document nodes.

        Yields:
            Tuple of (id, label, properties) for each document
        """
        for doc in self.corpus.documents:
            properties = {
                "name": doc.name or doc.id,
                "text": doc.text[:500] if doc.text else "",  # Truncate long text
            }
            # Add document metadata if available
            if doc.metadata:
                for key, value in doc.metadata.items():
                    # Skip large metadata items
                    if isinstance(value, (str, int, float, bool)):
                        properties[f"doc_{key}"] = value

            yield (doc.id, "document", properties)

    def _generate_keyword_nodes(self):
        """
        Generator for keyword nodes.

        Yields:
            Tuple of (id, label, properties) for each unique keyword
        """
        keywords = set()
        for doc in self.corpus.documents:
            if doc.metadata and "keywords" in doc.metadata:
                doc_keywords = doc.metadata["keywords"]
                if isinstance(doc_keywords, list):
                    keywords.update(doc_keywords)
                elif isinstance(doc_keywords, str):
                    keywords.update([kw.strip() for kw in doc_keywords.split(",")])

        for keyword in keywords:
            yield (f"keyword:{keyword}", "keyword", {"name": keyword})

    def _generate_cluster_nodes(self):
        """
        Generator for cluster nodes.

        Yields:
            Tuple of (id, label, properties) for each cluster
        """
        # Check if corpus has cluster metadata
        clusters = set()
        for doc in self.corpus.documents:
            if doc.metadata and "cluster" in doc.metadata:
                doc_cluster = doc.metadata["cluster"]
                if isinstance(doc_cluster, (str, int)):
                    clusters.add(doc_cluster)

        for cluster_id in clusters:
            yield (f"cluster:{cluster_id}", "cluster", {"name": cluster_id})

    def _generate_metadata_nodes(self):
        """
        Generator for metadata nodes from DataFrame.

        Yields:
            Tuple of (id, label, properties) for metadata fields
        """
        if self.corpus.df is not None and not self.corpus.df.empty:
            # Check if there's an ID column that aligns with documents
            df_columns = list(self.corpus.df.columns)

            # Look for ID or common identifier columns
            id_col = None
            for potential_id in ["id", "ID", "doc_id", "document_id", "index"]:
                if potential_id in df_columns:
                    id_col = potential_id
                    break

            if id_col:
                # Create metadata nodes for each row
                for idx, row in self.corpus.df.iterrows():
                    doc_id = row[id_col]
                    properties = {}
                    for col in df_columns:
                        if col != id_col:
                            value = row[col]
                            # Skip NaN values
                            if not pd.isna(value):
                                properties[col] = value

                    if properties:  # Only create node if there are properties
                        yield (f"metadata:{doc_id}", "metadata", properties)

    def _generate_document_keyword_edges(self):
        """
        Generator for edges connecting documents to keywords.

        Yields:
            Tuple of (source_id, target_id, label, properties) for each edge
        """
        for doc in self.corpus.documents:
            if doc.metadata and "keywords" in doc.metadata:
                doc_keywords = doc.metadata["keywords"]
                keywords_list = []

                if isinstance(doc_keywords, list):
                    keywords_list = doc_keywords
                elif isinstance(doc_keywords, str):
                    keywords_list = [kw.strip() for kw in doc_keywords.split(",")]

                for keyword in keywords_list:
                    yield (doc.id, f"keyword:{keyword}", "HAS_KEYWORD", {})

    def _generate_document_cluster_edges(self):
        """
        Generator for edges connecting documents to clusters.

        Yields:
            Tuple of (source_id, target_id, label, properties) for each edge
        """
        # Check if documents have cluster assignments
        for doc in self.corpus.documents:
            if doc.metadata and "cluster" in doc.metadata:
                cluster_id = doc.metadata["cluster"]
                yield (doc.id, f"cluster:{cluster_id}", "BELONGS_TO_CLUSTER", {})

    def _generate_document_metadata_edges(self):
        """
        Generator for edges connecting documents to metadata.

        Yields:
            Tuple of (source_id, target_id, label, properties) for each edge
        """
        if self.corpus.df is not None and not self.corpus.df.empty:
            df_columns = list(self.corpus.df.columns)

            # Look for ID column
            id_col = None
            for potential_id in ["id", "ID", "doc_id", "document_id", "index"]:
                if potential_id in df_columns:
                    id_col = potential_id
                    break

            if id_col:
                for idx, row in self.corpus.df.iterrows():
                    doc_id = row[id_col]
                    yield (doc_id, f"metadata:{doc_id}", "HAS_METADATA", {})

    def _generate_cluster_keyword_edges(self):
        """
        Generator for edges connecting clusters to keywords.

        Yields:
            Tuple of (source_id, target_id, label, properties) for each edge
        """
        # Collect all clusters and keywords
        clusters = set()
        keywords = set()
        for doc in self.corpus.documents:
            if doc.metadata:
                if "cluster" in doc.metadata:
                    clusters.add(doc.metadata["cluster"])
                if "keywords" in doc.metadata:
                    doc_keywords = doc.metadata["keywords"]
                    if isinstance(doc_keywords, list):
                        keywords.update(doc_keywords)
                    elif isinstance(doc_keywords, str):
                        keywords.update([kw.strip() for kw in doc_keywords.split(",")])
        # Create edges between every cluster and every keyword
        for cluster_id in clusters:
            for keyword in keywords:
                yield (
                    f"cluster:{cluster_id}",
                    f"keyword:{keyword}",
                    "CLUSTER_HAS_KEYWORD",
                    {},
                )

    def create_graph(self) -> Dict[str, Any]:
        """
        Create a graph representation of the corpus.

        This method creates nodes for documents, keywords, clusters, and metadata,
        and edges connecting them. The graph is stored in corpus metadata.

        Returns:
            Dictionary containing graph data (nodes and edges)

        Raises:
            ValueError: If documents don't have keywords assigned
        """
        # Check if documents have keywords
        has_keywords = False
        for doc in self.corpus.documents:
            if doc.metadata and "keywords" in doc.metadata:
                has_keywords = True
                break

        if not has_keywords:
            logger.error(
                "Documents do not have keywords assigned. Please run --assign before generating the graph."
            )
            raise ValueError(
                "Documents do not have keywords assigned. Run keyword assignment (--assign) first."
            )

        # Check if DataFrame can be included
        can_include_df = False
        if self.corpus.df is not None and not self.corpus.df.empty:
            df_columns = list(self.corpus.df.columns)
            for potential_id in ["id", "ID", "doc_id", "document_id", "index"]:
                if potential_id in df_columns:
                    can_include_df = True
                    break

            if not can_include_df:
                logger.warning(
                    "DataFrame does not have an aligning ID field. DataFrame metadata will not be included in the graph."
                )

        # Create NetworkX graph for internal representation
        self.graph = nx.Graph()

        # Collect all nodes and edges
        nodes = []
        edges = []

        # Add document nodes
        for node_data in self._generate_document_nodes():
            node_id, label, properties = node_data
            nodes.append({"id": node_id, "label": label, "properties": properties})
            self.graph.add_node(node_id, label=label, **properties)

        # Add keyword nodes
        for node_data in self._generate_keyword_nodes():
            node_id, label, properties = node_data
            nodes.append({"id": node_id, "label": label, "properties": properties})
            self.graph.add_node(node_id, label=label, **properties)

        # Add cluster nodes if present
        for node_data in self._generate_cluster_nodes():
            node_id, label, properties = node_data
            nodes.append({"id": node_id, "label": label, "properties": properties})
            self.graph.add_node(node_id, label=label, **properties)

        # Add metadata nodes if applicable
        if can_include_df:
            for node_data in self._generate_metadata_nodes():
                node_id, label, properties = node_data
                nodes.append({"id": node_id, "label": label, "properties": properties})
                self.graph.add_node(node_id, label=label, **properties)

        # Add document-keyword edges
        for edge_data in self._generate_document_keyword_edges():
            source_id, target_id, label, properties = edge_data
            edges.append(
                {
                    "source": source_id,
                    "target": target_id,
                    "label": label,
                    "properties": properties,
                }
            )
            self.graph.add_edge(source_id, target_id, label=label, **properties)

        # Add document-cluster edges if present
        for edge_data in self._generate_document_cluster_edges():
            source_id, target_id, label, properties = edge_data
            edges.append(
                {
                    "source": source_id,
                    "target": target_id,
                    "label": label,
                    "properties": properties,
                }
            )
            self.graph.add_edge(source_id, target_id, label=label, **properties)

        # Add document-metadata edges if applicable
        if can_include_df:
            for edge_data in self._generate_document_metadata_edges():
                source_id, target_id, label, properties = edge_data
                edges.append(
                    {
                        "source": source_id,
                        "target": target_id,
                        "label": label,
                        "properties": properties,
                    }
                )
                self.graph.add_edge(source_id, target_id, label=label, **properties)

        # Add cluster-keyword edges
        for edge_data in self._generate_cluster_keyword_edges():
            source_id, target_id, label, properties = edge_data
            edges.append(
                {
                    "source": source_id,
                    "target": target_id,
                    "label": label,
                    "properties": properties,
                }
            )
            self.graph.add_edge(source_id, target_id, label=label, **properties)

        # Add keyword-metadata edges
        for edge_data in self._generate_keyword_metadata_edges():
            source_id, target_id, label, properties = edge_data
            edges.append(
                {
                    "source": source_id,
                    "target": target_id,
                    "label": label,
                    "properties": properties,
                }
            )
            self.graph.add_edge(source_id, target_id, label=label, **properties)

        # Add cluster-metadata edges
        for edge_data in self._generate_cluster_metadata_edges():
            source_id, target_id, label, properties = edge_data
            edges.append(
                {
                    "source": source_id,
                    "target": target_id,
                    "label": label,
                    "properties": properties,
                }
            )
            self.graph.add_edge(source_id, target_id, label=label, **properties)

        # Create graph data structure
        graph_data = {
            "nodes": nodes,
            "edges": edges,
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "num_documents": len(self.corpus.documents),
            "has_keywords": has_keywords,
            "has_clusters": "clusters" in self.corpus.metadata,
            "has_metadata": can_include_df,
        }

        # Store in corpus metadata
        self.corpus.metadata["graph"] = graph_data

        logger.info(f"Graph created with {len(nodes)} nodes and {len(edges)} edges")

        return graph_data

    def get_networkx_graph(self) -> nx.Graph:
        """
        Get the NetworkX graph representation.

        Returns:
            NetworkX Graph object

        Raises:
            ValueError: If graph hasn't been created yet
        """
        if self.graph is None:
            raise ValueError("Graph not created yet. Call create_graph() first.")
        return self.graph
