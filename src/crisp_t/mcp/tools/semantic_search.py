"""
Semantic Search Tools for MCP Server

This module contains tools for semantic search using ChromaDB embeddings.
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


def get_semantic_search_tools() -> list[Tool]:
    """Return list of semantic search tool definitions."""
    return [
        Tool(
            name="semantic_search",
            description="Perform semantic search to find documents similar to query using ChromaDB embeddings (not keyword search). Essential for: Literature reviews (find relevant papers), Qualitative coding (find documents matching themes), Exploring corpus conceptually. Returns documents ordered by semantic similarity (highest first). Query examples: 'healthcare barriers' (finds relevant passages regardless of exact wording). Compare to: filter_documents (exact metadata matching), semantic_chunk_search (searches within one document). Workflow: semantic_search(query='theme') → examine results → use top doc IDs for find_similar_documents. Tip: Refine query if results miss relevant documents.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text (conceptual search, not keyword)",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of similar documents to return (default: 5, typical: 3-20)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="find_similar_documents",
            description="Find documents semantically similar to reference documents (seed-based search). Essential for: Literature review snowballing (start with known relevant papers → find more similar ones), Validation (are these documents similar to my key examples?), Grouping (find all docs similar to this cluster). Parameters: document_ids (one or multiple), n_results (documents to return), threshold (minimum similarity 0-1, default 0.7 = high similarity). Workflow: Pick 1-3 exemplary documents → find_similar_documents → expand literature set → validate quality. Typical threshold: 0.5 (loose matching) to 0.9 (strict matching). Compare to: semantic_search (query-based), semantic_chunk_search (within-document search). Tip: Start with threshold=0.5 if getting too few results.",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_ids": {
                        "type": "string",
                        "description": "Single document ID or comma-separated list of IDs to use as reference",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of similar documents to return (default: 5, typical: 3-20)",
                        "default": 5,
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Minimum similarity threshold 0-1 (default: 0.7=high, try 0.5 for loose matching, 0.9 for strict)",
                        "default": 0.7,
                    },
                },
                "required": ["document_ids"],
            },
        ),
        Tool(
            name="semantic_chunk_search",
            description="Perform semantic search within a single document to find relevant passages/chunks matching a concept or theme. Essential for: Qualitative coding (find mentions of concept X in document Y), Annotation (mark relevant sections for later analysis), Quote extraction (identify key passages supporting theme). Parameters: query (concept/theme), doc_id (document to search), threshold (minimum similarity 0-1, default 0.5), n_results (max chunks to retrieve). Workflow: Get doc_id with semantic_search → semantic_chunk_search(query='theme', doc_id=X) → extract matching chunks for coding/quotes. Threshold guidance: 0.3-0.5 (loose, catch all mentions), 0.7+ (strict, only high matches). Compare to: semantic_search (whole corpus), filter_documents (metadata-based). Tip: Use for manual annotation/coding validation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (concept/theme to find within document)",
                    },
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID to search within",
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Minimum similarity threshold 0-1 (default: 0.5; try 0.3-0.7 range)",
                        "default": 0.5,
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Maximum chunks to retrieve before filtering (default: 10, typical: 5-20)",
                        "default": 10,
                    },
                },
                "required": ["query", "doc_id"],
            },
        ),
        Tool(
            name="export_metadata_df",
            description="Export ChromaDB collection metadata as DataFrame for analysis/visualization. Essential for: Analyzing document-level metadata (sources, dates, etc.), Exporting for external analysis tools, Creating reports. Parameters: metadata_keys (comma-separated keys to export, optional - all keys by default). Returns: DataFrame with document metadata. Workflow: semantic_search → export_metadata_df(metadata_keys='source,date') → analyze patterns in metadata.",
            inputSchema={
                "type": "object",
                "properties": {
                    "metadata_keys": {
                        "type": "string",
                        "description": "Comma-separated metadata keys to include (optional, all if not specified)",
                    },
                },
            },
        ),
    ]


def handle_semantic_search_tool(
    name: str,
    arguments: dict[str, Any],
    corpus: Any,
    text_analyzer: Any,
    csv_analyzer: Any,
    ml_analyzer: Any,
) -> tuple[list[TextContent], Any, Any] | None:
    """Handle semantic search tool calls.
    
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
    if name == "semantic_search":
        if not corpus:
            return no_corpus_response(), corpus, ml_analyzer

        try:
            from ...semantic import Semantic

            query = arguments.get("query")
            if not query:
                return error_response("query is required"), corpus, ml_analyzer

            n_results = arguments.get("n_results", 5)

            semantic_analyzer = Semantic(corpus)
            result_corpus = semantic_analyzer.get_similar(
                query, n_results=n_results
            )

            # Prepare response
            response_text = f"Semantic search completed for query: '{query}'\n"
            response_text += (
                f"Found {len(result_corpus.documents)} similar documents\n\n"
            )
            response_text += "Document IDs:\n"
            for doc in result_corpus.documents[:10]:  # Show first 10
                response_text += f"- {doc.id}: {doc.name or 'No name'}\n"
            if len(result_corpus.documents) > 10:
                response_text += (
                    f"... and {len(result_corpus.documents) - 10} more\n"
                )

            return success_response(response_text), result_corpus, ml_analyzer

        except ImportError:
            return (
                error_response(
                    "chromadb is not installed. Install with: pip install chromadb"
                ),
                corpus,
                ml_analyzer,
            )
        except Exception as e:
            return (
                error_response(f"Error during semantic search: {e}"),
                corpus,
                ml_analyzer,
            )

    elif name == "find_similar_documents":
        if not corpus:
            return no_corpus_response(), corpus, ml_analyzer

        try:
            from ...semantic import Semantic

            document_ids = arguments.get("document_ids")
            if not document_ids:
                return (
                    error_response("document_ids is required"),
                    corpus,
                    ml_analyzer,
                )

            n_results = arguments.get("n_results", 5)
            threshold = arguments.get("threshold", 0.7)

            semantic_analyzer = Semantic(corpus)
            similar_doc_ids = semantic_analyzer.get_similar_documents(
                document_ids=document_ids, n_results=n_results, threshold=threshold
            )

            # Prepare response
            response_text = f"Finding documents similar to: '{document_ids}'\n"
            response_text += f"Number of results requested: {n_results}\n"
            response_text += f"Similarity threshold: {threshold}\n"
            response_text += f"Found {len(similar_doc_ids)} similar documents\n\n"

            if similar_doc_ids:
                response_text += "Similar Document IDs:\n"
                for doc_id in similar_doc_ids:
                    doc = corpus.get_document_by_id(doc_id)
                    doc_name = f" - {doc.name}" if doc and doc.name else ""
                    response_text += f"  • {doc_id}{doc_name}\n"

                response_text += "\nThis feature is useful for:\n"
                response_text += (
                    "- Literature reviews: Find additional relevant papers\n"
                )
                response_text += "- Qualitative research: Identify documents with similar themes\n"
                response_text += (
                    "- Content grouping: Group similar documents for analysis\n"
                )
            else:
                response_text += "No similar documents found above the threshold.\n"
                response_text += "Try lowering the threshold or using different reference documents.\n"

            return success_response(response_text), corpus, ml_analyzer

        except ImportError:
            return (
                error_response(
                    "chromadb is not installed. Install with: pip install chromadb"
                ),
                corpus,
                ml_analyzer,
            )
        except Exception as e:
            return (
                error_response(f"Error finding similar documents: {e}"),
                corpus,
                ml_analyzer,
            )

    elif name == "semantic_chunk_search":
        if not corpus:
            return no_corpus_response(), corpus, ml_analyzer

        try:
            from ...semantic import Semantic

            query = arguments.get("query")
            doc_id = arguments.get("doc_id")

            if not query:
                return error_response("query is required"), corpus, ml_analyzer
            if not doc_id:
                return error_response("doc_id is required"), corpus, ml_analyzer

            threshold = arguments.get("threshold", 0.5)
            n_results = arguments.get("n_results", 10)

            semantic_analyzer = Semantic(corpus)
            chunks = semantic_analyzer.get_similar_chunks(
                query=query, doc_id=doc_id, threshold=threshold, n_results=n_results
            )

            # Prepare response
            response_text = (
                f"Semantic chunk search completed for query: '{query}'\n"
            )
            response_text += f"Document ID: {doc_id}\n"
            response_text += f"Threshold: {threshold}\n"
            response_text += f"Found {len(chunks)} matching chunks\n\n"

            if chunks:
                response_text += "Matching chunks:\n"
                response_text += "=" * 60 + "\n\n"
                for i, chunk in enumerate(chunks, 1):
                    response_text += f"Chunk {i}:\n"
                    response_text += chunk + "\n"
                    response_text += "-" * 60 + "\n\n"

                response_text += f"\nThese {len(chunks)} chunks can be used for coding/annotating the document.\n"
                response_text += (
                    "You can adjust the threshold to get more or fewer results.\n"
                )
            else:
                response_text += (
                    "No chunks matched the query above the threshold.\n"
                )
                response_text += (
                    "Try lowering the threshold or use a different query.\n"
                )

            return success_response(response_text), corpus, ml_analyzer

        except ImportError:
            return (
                error_response(
                    "chromadb is not installed. Install with: pip install chromadb"
                ),
                corpus,
                ml_analyzer,
            )
        except Exception as e:
            return (
                error_response(f"Error during semantic chunk search: {e}"),
                corpus,
                ml_analyzer,
            )

    elif name == "export_metadata_df":
        if not corpus:
            return (
                [
                    TextContent(
                        type="text", text="No corpus loaded. Use load_corpus first."
                    )
                ],
                corpus,
                ml_analyzer,
            )

        try:
            from ...semantic import Semantic

            metadata_keys_str = arguments.get("metadata_keys")
            metadata_keys = None
            if metadata_keys_str:
                metadata_keys = [k.strip() for k in metadata_keys_str.split(",")]

            semantic_analyzer = Semantic(corpus)
            result_corpus = semantic_analyzer.get_df(metadata_keys=metadata_keys)

            # Prepare response
            if result_corpus.df is not None:
                response_text = "Metadata exported to DataFrame\n"
                response_text += f"Shape: {result_corpus.df.shape}\n"
                response_text += f"Columns: {list(result_corpus.df.columns)}\n\n"
                response_text += "First 5 rows:\n"
                response_text += result_corpus.df.head().to_string()
                return success_response(response_text), result_corpus, ml_analyzer
            else:
                return (
                    error_response("No DataFrame created"),
                    corpus,
                    ml_analyzer,
                )

        except ImportError:
            return (
                error_response(
                    "chromadb is not installed. Install with: pip install chromadb"
                ),
                corpus,
                ml_analyzer,
            )
        except Exception as e:
            return (
                error_response(f"Error exporting metadata: {e}"),
                corpus,
                ml_analyzer,
            )

    # Tool not handled by this module
    return None
