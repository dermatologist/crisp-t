"""
Miscellaneous Tools for MCP Server

This module contains miscellaneous utility tools for managing server state and cache.
"""

import logging
from typing import Any

from mcp.types import TextContent, Tool

from ...helpers.clib import clear_cache
from ..utils.responses import (
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)


def get_misc_tools() -> list[Tool]:
    """Return list of miscellaneous tool definitions."""
    return [
        Tool(
            name="reset_corpus_state",
            description="Reset the global corpus, text analyzer, and CSV analyzer state. Clear all loaded data and start fresh.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="clear_cache",
            description="Delete the cache folder if it exists. Use this to clear cached analysis results and free up disk space.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


def handle_misc_tool(
    name: str,
    arguments: dict[str, Any],
    corpus: Any,
    text_analyzer: Any,
    csv_analyzer: Any,
    ml_analyzer: Any,
) -> tuple[list[TextContent], Any, Any, Any, Any] | None:
    """Handle miscellaneous tool calls.
    
    Args:
        name: Tool name
        arguments: Tool arguments
        corpus: Current corpus
        text_analyzer: Current text analyzer
        csv_analyzer: Current CSV analyzer
        ml_analyzer: Current ML analyzer
        
    Returns:
        Tuple of (response, updated_corpus, updated_text_analyzer, updated_csv_analyzer, updated_ml_analyzer) or None if tool not handled
    """
    if name == "reset_corpus_state":
        corpus = None
        text_analyzer = None
        csv_analyzer = None
        ml_analyzer = None
        return success_response("Global corpus state has been reset."), corpus, text_analyzer, csv_analyzer, ml_analyzer

    elif name == "clear_cache":
        try:
            clear_cache()
            return success_response("Cache has been cleared successfully."), corpus, text_analyzer, csv_analyzer, ml_analyzer
        except Exception as e:
            return error_response(f"Error clearing cache: {str(e)}"), corpus, text_analyzer, csv_analyzer, ml_analyzer

    # Tool not handled by this module
    return None
