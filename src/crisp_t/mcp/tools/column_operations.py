"""
CSV Column/DataFrame Operations Tools for MCP Server

This module contains tools for DataFrame column operations including binning,
encoding, filtering, oversampling, and column type operations.
"""

import json
import logging
from typing import Any

from mcp.types import TextContent, Tool

from ..utils.responses import (
    no_csv_analyzer_response,
    success_response,
)

logger = logging.getLogger(__name__)


def get_column_operations_tools() -> list[Tool]:
    """Return list of column operations tool definitions."""
    return [
        Tool(
            name="bin_a_column",
            description="Convert numeric column to categorical by binning into equal-width intervals. Essential preprocessing for: Creating outcome categories (e.g., satisfaction_score → low/medium/high), Preparing data for categorical ML algorithms, Creating linked text↔numeric relationships. Bins parameter: Default 2 (binary), 3-5 common for typical ranges, 10+ for fine-grained analysis. Workflow: get_column_values → bin_a_column (default 3 bins) → one_hot_encode_column → use in ML. Tip: Inspect distribution first with get_df_row to choose appropriate bin count.",
            inputSchema={
                "type": "object",
                "properties": {
                    "column_name": {
                        "type": "string",
                        "description": "Name of the numeric column to bin",
                    },
                    "bins": {
                        "type": "integer",
                        "description": "Number of bins (default: 2, typical: 3-5)",
                        "default": 2,
                    },
                },
                "required": ["column_name"],
            },
        ),
        Tool(
            name="one_hot_encode_column",
            description="One-hot encode categorical column to binary indicator columns (necessary preprocessing for many ML algorithms). Essential for: Tree-based models (decision_tree_classification, random forests), Linear models (regression_analysis), Neural networks. Creates dummy columns for each category. Workflow: After bin_a_column or for natural categorical columns → one_hot_encode_column → train_model. Tip: Ensure categorical column is properly formatted (strings or ints); numbers are not auto-detected as categories unless binned first. Tip: For multiclass problems, removes one redundant column to prevent multicollinearity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "column_name": {
                        "type": "string",
                        "description": "Name of the column to one-hot encode",
                    }
                },
                "required": ["column_name"],
            },
        ),
        Tool(
            name="filter_rows_by_column_value",
            description="Filter DataFrame to keep only rows matching specific column value. Essential for: Subsetting data (e.g., keep only 'Treatment' group), Data exploration (examine specific categories), Linked analysis (combine with filter_documents for text+numeric matching). Returns filtered DataFrame. Workflow: get_df_columns → get_column_values(column) to see options → filter_rows_by_column_value to subset. Note: Supports both string and numeric values (auto-detected). Tip: Use with filter_documents(metadata_filter=...) to coordinate text and numeric subsetting.",
            inputSchema={
                "type": "object",
                "properties": {
                    "column_name": {
                        "type": "string",
                        "description": "Column to filter on",
                    },
                    "value": {
                        "type": "string",
                        "description": "Value to match (numeric values are auto-detected)",
                    },
                },
                "required": ["column_name", "value"],
            },
        ),
        Tool(
            name="oversample",
            description="Apply random oversampling to balance imbalanced classes in DataFrame. Essential for: Imbalanced classification (rare outcome prediction), Ensuring model trains on balanced proportions. Workflow: Prepare X/y via ML tools → Check class distribution (get_column_values) → oversample if imbalanced (target: ~50/50 or equal proportions) → train_model. Use restore_oversample after model training to return to original data. Warning: Oversampling increases data size and can slow training; best for small datasets (<10K rows). Tip: For large datasets, consider stratified sampling or class weights in ML algorithm instead.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="restore_oversample",
            description="Restore X and y to original (pre-oversampling) values. Use after model training when oversampling was applied. Workflow: oversample → train_model → evaluate_model → restore_oversample to return to original proportions for final evaluation on unbalanced data. Essential for: Accurate performance metrics on real-world imbalanced data, Preventing overfitting from synthetic duplicates, Final validation of model generalization.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_column_types",
            description="Get data types of all DataFrame columns. Essential for: Understanding data format (int64, float64, object/string, etc.), Validating data after loading/preprocessing, Planning feature engineering (e.g., categorical vs numeric columns). Workflow: load data → get_df_columns → get_column_types to understand structure → plan preprocessing. Use retain_numeric_columns_only if you need only numeric features for ML.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_column_values",
            description="Get all unique values from specific DataFrame column with value counts. Essential for: Exploring column contents before filtering/analysis, Understanding categorical distributions, Planning binning (get_column_values before bin_a_column to understand numeric range), Validation after preprocessing. Workflow: get_df_columns → get_column_values(column) → filter_rows_by_column_value (with desired value). Returns unique values and their frequencies (if available). Useful for both numeric and categorical columns.",
            inputSchema={
                "type": "object",
                "properties": {
                    "column_name": {
                        "type": "string",
                        "description": "Column name to retrieve values from",
                    }
                },
                "required": ["column_name"],
            },
        ),
        Tool(
            name="retain_numeric_columns_only",
            description="Keep only numeric columns in DataFrame; remove string/object columns. Essential preprocessing for: Preparing data for ML algorithms (most require numeric input), PCA analysis, Regression/classification with numeric features. Workflow: get_column_types → identify categorical columns (if needed: one_hot_encode first) → retain_numeric_columns_only → use in ML model. Warning: This removes all non-numeric data permanently from active corpus; consider encoding categoricals to numeric before using. Tip: Compare with get_column_types first to understand what will be removed.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


def handle_column_operations_tool(
    name: str,
    arguments: dict[str, Any],
    csv_analyzer: Any,
) -> list[TextContent] | None:
    """Handle column operations tool calls.
    
    Args:
        name: Tool name
        arguments: Tool arguments
        csv_analyzer: Current CSV analyzer
        
    Returns:
        Response as list of TextContent or None if tool not handled
    """
    if name == "bin_a_column":
        if not csv_analyzer:
            return no_csv_analyzer_response()

        msg = csv_analyzer.bin_a_column(
            column_name=arguments["column_name"], bins=arguments.get("bins", 2)
        )
        return success_response(str(msg))

    elif name == "one_hot_encode_column":
        if not csv_analyzer:
            return no_csv_analyzer_response()

        msg = csv_analyzer.one_hot_encode_column(
            column_name=arguments["column_name"]
        )
        return success_response(str(msg))

    elif name == "filter_rows_by_column_value":
        if not csv_analyzer:
            return no_csv_analyzer_response()

        msg = csv_analyzer.filter_rows_by_column_value(
            column_name=arguments["column_name"], value=arguments["value"], mcp=True
        )
        return success_response(str(msg))

    elif name == "oversample":
        if not csv_analyzer:
            return no_csv_analyzer_response()

        result = csv_analyzer.oversample(mcp=True)
        return success_response(str(result))

    elif name == "restore_oversample":
        if not csv_analyzer:
            return no_csv_analyzer_response()

        result = csv_analyzer.restore_oversample(mcp=True)
        return success_response(str(result))

    elif name == "get_column_types":
        if not csv_analyzer:
            return no_csv_analyzer_response()

        types = csv_analyzer.get_column_types()
        return success_response(json.dumps(types, indent=2, default=str))

    elif name == "get_column_values":
        if not csv_analyzer:
            return no_csv_analyzer_response()

        values = csv_analyzer.get_column_values(arguments["column_name"])
        return success_response(json.dumps(values, indent=2, default=str))

    elif name == "retain_numeric_columns_only":
        if not csv_analyzer:
            return no_csv_analyzer_response()

        csv_analyzer.retain_numeric_columns_only()
        return success_response("Retained numeric columns only.")

    # Tool not handled by this module
    return None
