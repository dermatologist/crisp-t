"""
Corpus Filtering Tools for MCP Server

This module contains tools for filtering corpus documents and managing document counts.
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


def get_corpus_filtering_tools() -> list[Tool]:
    """Return list of corpus filtering tool definitions."""
    return [
        Tool(
            name="filter_documents",
            description=(
                "Filter corpus documents based on coding links and/or metadata filters. "
                "Supports metadata filters (key:value or key=value) and link filters for embedding and temporal relationships. "
                "Link format: 'embedding:text' (documents with embedding links), 'embedding:df' (matches temporal links), etc. "
                "Apply AND logic when combining filters. Returns filtered corpus and document count. "
                "Use to subset corpus for sub-analysis, identify documents with specific relationships, or validate link creation. "
                "Tip: Use list_documents first to understand existing links before filtering. "
                "Workflow: Filter to documents with specific coded categories → analyze subset → save as branch for comparison. "
                "Updates the active corpus."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "string",
                        "description": (
                            "Filter string. Examples: 'keywords=health', 'embedding:text', 'embedding:df', 'temporal:text', 'temporal:df'. "
                            "Multiple filters can be applied sequentially."
                        ),
                    },
                },
                "required": ["filter"],
            },
        ),
        Tool(
            name="document_count",
            description="Return number of documents currently in active corpus (accounting for any active filters). Use to validate filtering operations or understand corpus size after subsetting. Useful in workflows: load corpus → filter by criteria → check count to verify expected subset size.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


def handle_corpus_filtering_tool(
    name: str,
    arguments: dict[str, Any],
    corpus: Any,
    text_analyzer: Any,
    csv_analyzer: Any,
    ml_analyzer: Any,
) -> tuple[list[TextContent], Any, Any] | None:
    """Handle corpus filtering tool calls.
    
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
    if name == "filter_documents":
        if not text_analyzer:
            return no_corpus_response(), corpus, ml_analyzer

        metadata_key = arguments.get("metadata_key", "keywords")
        metadata_value = arguments.get("metadata_value")
        if not metadata_value:
            return error_response("metadata_value is required"), corpus, ml_analyzer

        msg = text_analyzer.filter_documents(
            metadata_key=metadata_key, metadata_value=metadata_value, mcp=True
        )
        return success_response(str(msg)), corpus, ml_analyzer

    elif name == "document_count":
        if not text_analyzer:
            return no_corpus_response(), corpus, ml_analyzer

        try:
            count = text_analyzer.document_count()
        except Exception as e:
            return error_response(str(e)), corpus, ml_analyzer
        return success_response(f"Document count: {count}"), corpus, ml_analyzer

    # Tool not handled by this module
    return None
