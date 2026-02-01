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

from ..utils.responses import error_response, no_corpus_response, success_response


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
        Tool(
            name="get_df_shape",
            description="Get DataFrame dimensions (rows, columns). Provides quick overview of dataset size. Essential for: Understanding data scale before analysis, Validating transformations (compare before/after), Checking memory requirements for ML. Workflow: get_df_shape → note dimensions → apply transformations → get_df_shape again to verify. Returns tuple (n_rows, n_columns).",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="mark_missing",
            description="Remove rows containing empty strings or NaN values from DataFrame. Essential data cleaning step for: Removing incomplete records before analysis, Ensuring ML models don't encounter missing data, Improving data quality. Permanently modifies the DataFrame. Workflow: get_df_row_count (before) → mark_missing → get_df_row_count (after) to see how many rows removed. Tip: Use before ML training to avoid errors.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="mark_duplicates",
            description="Remove duplicate rows from DataFrame based on all columns. Essential for: Data deduplication before analysis, Ensuring unique records for statistical validity, Preventing bias from repeated entries. Permanently modifies the DataFrame. Workflow: get_df_row_count (before) → mark_duplicates → get_df_row_count (after) to see duplicates removed. Tip: Run after data import to ensure clean dataset.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="restore_df",
            description="Restore DataFrame to its original state (before any transformations). Useful for: Undoing transformations and starting fresh, Comparing original vs. transformed data, Recovering from errors. Workflow: Load data → make transformations → if unsatisfied → restore_df → try different approach. Tip: This cannot be undone - any changes after restore are permanent.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="drop_na",
            description="Remove all rows with any NA/NaN values from DataFrame. More aggressive than mark_missing - removes any row with even a single NA value. Essential for: Strict data quality requirements, Preparing for algorithms that don't handle NaN, Ensuring complete cases only. Workflow: get_df_row_count → drop_na → get_df_row_count to see impact. Warning: Can significantly reduce dataset size if many columns have scattered NAs.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


from typing import Optional


def handle_dataframe_operations_tool(
    name: str,
    arguments: Dict[str, Any],
    corpus: Any,
    text_analyzer: Any = None,
    csv_analyzer: Any = None,
    ml_analyzer: Any = None,
) -> Optional[tuple[list[Any], Any, Any]]:
    """Handle DataFrame operations tool calls.

    Args:
        name: Tool name
        arguments: Tool arguments
        corpus: Corpus instance
        text_analyzer: Text analyzer instance
        csv_analyzer: CSV analyzer instance
        ml_analyzer: ML analyzer instance

    Returns:
        Tuple of (response, updated_corpus, updated_ml_analyzer) or None
    """
    from ..utils.responses import no_csv_analyzer_response
    
    if name == "get_df_columns":
        if not corpus:
            return no_corpus_response(), corpus, None

        cols = corpus.get_all_df_column_names()
        return success_response(json.dumps(cols, indent=2)), corpus, None

    elif name == "get_df_row_count":
        if not corpus:
            return no_corpus_response(), corpus, None

        count = corpus.get_row_count()
        return success_response(f"Row count: {count}"), corpus, None

    elif name == "get_df_row":
        if not corpus:
            return no_corpus_response(), corpus, None

        row = corpus.get_row_by_index(arguments["index"])
        if row is not None:
            return (
                success_response(json.dumps(row.to_dict(), indent=2, default=str)),
                corpus,
                None,
            )
        return error_response("Row not found"), corpus, None

    elif name == "get_df_shape":
        if not csv_analyzer:
            return no_csv_analyzer_response(), corpus, None
        
        shape = csv_analyzer.get_shape()
        return success_response(f"DataFrame shape: {shape[0]} rows × {shape[1]} columns"), corpus, None

    elif name == "mark_missing":
        if not csv_analyzer:
            return no_csv_analyzer_response(), corpus, None
        
        csv_analyzer.mark_missing()
        new_count = csv_analyzer.df.shape[0]
        return success_response(f"Removed rows with missing values. New row count: {new_count}"), corpus, None

    elif name == "mark_duplicates":
        if not csv_analyzer:
            return no_csv_analyzer_response(), corpus, None
        
        csv_analyzer.mark_duplicates()
        new_count = csv_analyzer.df.shape[0]
        return success_response(f"Removed duplicate rows. New row count: {new_count}"), corpus, None

    elif name == "restore_df":
        if not csv_analyzer:
            return no_csv_analyzer_response(), corpus, None
        
        csv_analyzer.restore_df()
        return success_response("DataFrame restored to original state"), corpus, None

    elif name == "drop_na":
        if not csv_analyzer:
            return no_csv_analyzer_response(), corpus, None
        
        csv_analyzer.drop_na()
        new_count = csv_analyzer.df.shape[0]
        return success_response(f"Removed all rows with NA values. New row count: {new_count}"), corpus, None

    return None
