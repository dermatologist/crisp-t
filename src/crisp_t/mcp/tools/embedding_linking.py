"""
Embedding-based Linking Tools for MCP Server

This module contains tools for linking documents to dataframe rows using
semantic embedding similarity.
"""

import logging
from typing import Any

from mcp.types import TextContent, Tool

from ..utils.responses import (
    error_response,
    no_corpus_response,
    success_response,
)

logger = logging.getLogger(__name__)


def get_embedding_linking_tools() -> list[Tool]:
    """Return list of embedding-based linking tool definitions."""
    return [
        Tool(
            name="embedding_link",
            description="""
            Link documents to dataframe rows using semantic embedding similarity. Essential for: Mixed-methods triangulation when explicit IDs/timestamps unavailable, Fuzzy matching based on content meaning, Validating qualitative themes against numeric patterns.

            Uses vector embeddings to create text↔numeric relationships based on semantic similarity (how conceptually similar are they?).

            Parameters:
            - similarity_metric: 'cosine' (default, most common for embeddings) or 'euclidean'
            - top_k: Number of DataFrame rows to link per document (default: 1, try 1-5)
            - threshold: Minimum similarity score 0-1 (e.g., 0.7 = high similarity; optional, filters low matches)
            - numeric_columns: Specific columns for embedding (default: all numeric)

            Workflow: Load corpus with numeric data → embedding_link(threshold=0.7, top_k=1) → get_relationships to inspect → visualize links.

            Compare to: temporal_link_by_time (uses timestamps), filter_documents (uses metadata/links).
            Tip: Start with top_k=1 and adjust based on results. Higher threshold=stricter matching.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "similarity_metric": {
                        "type": "string",
                        "description": "Similarity metric: 'cosine' (default) or 'euclidean'",
                        "enum": ["cosine", "euclidean"],
                        "default": "cosine",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top similar rows to link per document (default: 1, try 1-5)",
                        "default": 1,
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Minimum similarity threshold 0-1 (e.g., 0.7 for high; optional, filters low matches)",
                    },
                    "numeric_columns": {
                        "type": "string",
                        "description": "Comma-separated numeric columns for embeddings (optional, default: all numeric)",
                    },
                },
            },
        ),
        Tool(
            name="embedding_link_stats",
            description="""
            Get statistics about embedding-based links already created. Shows: document count linked, average similarity scores, distribution of links. Essential for: Validating embedding_link results, Understanding link quality (similarity score ranges), Reporting linking coverage.

            Returns: Number of linked documents, linking statistics (average similarity, etc.).

            Workflow: embedding_link(...) → embedding_link_stats → review statistics → if poor quality, re-run with different threshold/top_k → get_relationships for detailed links.
            Tip: Average similarity scores show link quality (0.7+ = good semantic alignment).
            """,
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


def handle_embedding_linking_tool(
    name: str,
    arguments: dict[str, Any],
    corpus: Any,
    text_analyzer: Any,
    csv_analyzer: Any,
    ml_analyzer: Any,
) -> tuple[list[TextContent], Any, Any] | None:
    """Handle embedding-based linking tool calls.
    
    Args:
        name: Tool name
        arguments: Tool arguments
        corpus: Current corpus
        text_analyzer: Current text analyzer
        csv_analyzer: Current CSV analyzer
        ml_analyzer: Current ML analyzer
        
    Returns:
        Tuple of (response, updated_corpus, updated_ml_analyzer) or None if tool not handled
    """
    if name == "embedding_link":
        if not corpus:
            return no_corpus_response(), corpus, ml_analyzer

        try:
            from ...embedding_linker import EmbeddingLinker

            similarity_metric = arguments.get("similarity_metric", "cosine")
            top_k = arguments.get("top_k", 1)
            threshold = arguments.get("threshold")
            numeric_columns_str = arguments.get("numeric_columns")

            numeric_columns = None
            if numeric_columns_str:
                numeric_columns = [
                    c.strip() for c in numeric_columns_str.split(",")
                ]

            linker = EmbeddingLinker(
                corpus,
                similarity_metric=similarity_metric,
                use_simple_embeddings=True,
            )
            corpus = linker.link_by_embedding_similarity(
                numeric_columns=numeric_columns, threshold=threshold, top_k=top_k
            )

            stats = linker.get_link_statistics()
            response_text = "Embedding-based linking complete\n\n"
            response_text += f"Linked documents: {stats['linked_documents']}/{stats['total_documents']}\n"
            response_text += f"Total links: {stats['total_links']}\n"
            response_text += f"Average similarity: {stats['avg_similarity']:.3f}\n"
            response_text += f"Similarity metric: {similarity_metric}\n"

            return success_response(response_text), corpus, ml_analyzer

        except ImportError:
            return error_response(
                "ChromaDB is not installed. Install with: pip install chromadb"
            ), corpus, ml_analyzer
        except Exception as e:
            return error_response(f"Error in embedding linking: {e}"), corpus, ml_analyzer

    elif name == "embedding_link_stats":
        if not corpus:
            return no_corpus_response(), corpus, ml_analyzer

        try:
            from ...embedding_linker import EmbeddingLinker

            # Check if corpus has embedding links
            has_links = any(
                "embedding_links" in doc.metadata
                and doc.metadata["embedding_links"]
                for doc in corpus.documents
            )

            if not has_links:
                return error_response(
                    "No embedding links found. Run embedding_link first."
                ), corpus, ml_analyzer

            linker = EmbeddingLinker(corpus, use_simple_embeddings=True)
            stats = linker.get_link_statistics()

            response_text = "Embedding Link Statistics:\n\n"
            response_text += f"Total documents: {stats['total_documents']}\n"
            response_text += f"Linked documents: {stats['linked_documents']}\n"
            response_text += f"Total links: {stats['total_links']}\n"
            response_text += f"Average similarity: {stats['avg_similarity']:.3f}\n"
            response_text += f"Min similarity: {stats['min_similarity']:.3f}\n"
            response_text += f"Max similarity: {stats['max_similarity']:.3f}\n"

            return success_response(response_text), corpus, ml_analyzer

        except Exception as e:
            return error_response(f"Error getting embedding statistics: {e}"), corpus, ml_analyzer

    # Tool not handled by this module
    return None
