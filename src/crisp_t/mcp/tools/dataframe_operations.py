"""
DataFrame Operations Tools for MCP Server

This module provides tools for basic DataFrame operations including:
- Getting column names
- Getting row count
- Getting specific rows
"""

import json
from typing import Any, Dict, List

from mcp.types import Tool

from .utils.responses import error_response, no_corpus_response, success_response


def get_dataframe_operations_tools() -> List[Tool]:
    """Get list of DataFrame operations tools.
    
    Returns:
        List of Tool objects for DataFrame operations
    """
    return [
        Tool(
            name="get_df_columns",
            description="Get all column names from the DataFrame. Essential first step in data exploration to understand available features for ML analysis or numeric linking. Returns column list and data types. Workflow: get_df_columns → get_column_types → filter/prepare columns → use in analysis. Tip: Use get_column_values to preview column contents.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_df_row_count",
            description="Get number of rows in the DataFrame. Essential for understanding dataset size before ML analysis or statistical testing. Check row count after filtering/preprocessing to validate data transformations. Workflow: get_df_row_count (before) → bin_a_column/oversample → get_df_row_count (after) to verify changes. Tip: Compare with document_count for text↔numeric alignment.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_df_row",
            description="Get specific row from DataFrame by index. Use for: Inspecting individual records, debugging data issues, validating values, extracting quotes for embedding context. Workflow: get_df_row_count to find valid range → get_df_row(index=N) to examine. Tip: Use after filtering to verify filter correctness. Returns all column values for that row.",
            inputSchema={
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "description": "Row index (0-based)"}
                },
                "required": ["index"],
            },
        ),
    ]


def handle_dataframe_operations_tool(
    name: str, arguments: Dict[str, Any], _corpus: Any
) -> List[Any]:
    """Handle DataFrame operations tool calls.
    
    Args:
        name: Tool name
        arguments: Tool arguments
        _corpus: Corpus instance
        
    Returns:
        List containing response content
    """
    if name == "get_df_columns":
        if not _corpus:
            return no_corpus_response()

        cols = _corpus.get_all_df_column_names()
        return success_response(json.dumps(cols, indent=2))

    elif name == "get_df_row_count":
        if not _corpus:
            return no_corpus_response()

        count = _corpus.get_row_count()
        return success_response(f"Row count: {count}")

    elif name == "get_df_row":
        if not _corpus:
            return no_corpus_response()

        row = _corpus.get_row_by_index(arguments["index"])
        if row is not None:
            return success_response(
                json.dumps(row.to_dict(), indent=2, default=str)
            )
        return error_response("Row not found")

    return error_response(f"Unknown tool: {name}")
