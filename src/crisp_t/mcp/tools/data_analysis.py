"""
Data Analysis Tools for MCP Server

This module provides advanced data analysis tools including:
- Correlation analysis
- Statistical queries
- Pattern detection
- Data aggregation
"""

import json
import logging
from typing import Any

from mcp.types import TextContent, Tool

from ..utils.responses import (
    error_response,
    no_csv_analyzer_response,
    success_response,
)

logger = logging.getLogger(__name__)


def get_data_analysis_tools() -> list[Tool]:
    """Return list of data analysis tool definitions."""
    return [
        Tool(
            name="compute_correlation",
            description="""
            Compute correlation matrix for numeric columns using Pearson, Kendall, or Spearman methods. Essential for: 
            Understanding relationships between variables, Identifying multicollinearity before ML, Feature selection for modeling, 
            Hypothesis testing about variable relationships.
            
            Methods:
            - pearson (default): Linear relationships, assumes normal distribution
            - kendall: Non-parametric, robust to outliers, good for ordinal data
            - spearman: Non-parametric, detects monotonic relationships
            
            Workflow: get_df_columns → select numeric columns → compute_correlation → find_significant_correlations to filter.
            Tip: Use threshold parameter to show only correlations above certain strength (e.g., 0.5 for moderate+).
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "string",
                        "description": "Comma-separated column names. If not provided, uses all numeric columns",
                    },
                    "method": {
                        "type": "string",
                        "description": "Correlation method: 'pearson' (default), 'kendall', or 'spearman'",
                        "enum": ["pearson", "kendall", "spearman"],
                        "default": "pearson",
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Show only correlations with absolute value >= threshold (0.0-1.0). Default: 0.0 (show all)",
                        "default": 0.0,
                    },
                },
            },
        ),
        Tool(
            name="find_significant_correlations",
            description="""
            Find and rank significant correlations above a threshold. Returns pairs of variables with their correlation coefficients.
            Essential for: Discovering strong relationships quickly, Prioritizing variables for deeper analysis, 
            Identifying redundant features, Generating research hypotheses.
            
            Output: Sorted list of variable pairs with correlation coefficients (highest magnitude first).
            
            Workflow: find_significant_correlations(threshold=0.7) → examine top pairs → investigate relationships → 
            add_relationship to document findings.
            Tip: Start with threshold=0.5 (moderate correlation), increase to 0.7-0.8 for strong relationships only.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "string",
                        "description": "Comma-separated column names. If not provided, uses all numeric columns",
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Minimum absolute correlation coefficient (0.0-1.0). Default: 0.5",
                        "default": 0.5,
                    },
                    "method": {
                        "type": "string",
                        "description": "Correlation method: 'pearson' (default), 'kendall', or 'spearman'",
                        "enum": ["pearson", "kendall", "spearman"],
                        "default": "pearson",
                    },
                },
            },
        ),
        Tool(
            name="execute_query",
            description="""
            Execute dynamic pandas queries on DataFrame for complex filtering, grouping, and sorting operations.
            Supports SQL-like operations without SQL syntax.
            
            Supported operations:
            - filter: Boolean conditions (e.g., "age > 30 and score < 50")
            - groupby: Aggregate by categories (e.g., "groupby:category agg:mean")
            - sort: Sort by columns (e.g., "sort:age:desc")
            - head/tail: Get top/bottom N rows (e.g., "head:10")
            
            Examples:
            - "age > 25 and status == 'active'" - Filter rows
            - "groupby:department agg:mean" - Average by department
            - "sort:salary:desc" - Sort by salary descending
            
            Workflow: Explore data → construct query → execute_query → save_result if needed.
            Tip: Use save_result=true to persist query results in corpus DataFrame.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Pandas query string (filter, groupby, sort, head, tail)",
                    },
                    "save_result": {
                        "type": "boolean",
                        "description": "Whether to save query result as new DataFrame (default: false)",
                        "default": False,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_column_statistics",
            description="""
            Get comprehensive statistical summary for specified columns including count, mean, std, min, quartiles, max.
            Essential for: Understanding data distributions, Detecting outliers, Validating data quality, 
            Planning transformations (scaling, binning).
            
            Returns: Statistical summary with measures of central tendency and spread for numeric columns.
            
            Workflow: get_df_columns → select columns of interest → get_column_statistics → 
            use insights to decide on preprocessing steps.
            Tip: Compare statistics before/after transformations to verify expected changes.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "string",
                        "description": "Comma-separated column names. If not provided, analyzes all numeric columns",
                    },
                },
            },
        ),
        Tool(
            name="get_unique_values_summary",
            description="""
            Get unique value counts for multiple columns to understand categorical data distribution.
            Essential for: Identifying categorical vs. continuous features, Detecting high-cardinality columns, 
            Planning encoding strategies, Understanding data diversity.
            
            Returns: For each column: number of unique values, top values with counts, percentage of unique values.
            
            Workflow: get_df_columns → get_unique_values_summary → identify high-cardinality columns → 
            decide encoding approach (one-hot for low cardinality, target encoding for high).
            Tip: Columns with >50% unique values may need special handling or should be treated as continuous.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "string",
                        "description": "Comma-separated column names. If not provided, analyzes all columns",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top values to show per column (default: 10)",
                        "default": 10,
                    },
                },
            },
        ),
    ]


def handle_data_analysis_tool(
    name: str,
    arguments: dict[str, Any],
    corpus: Any,
    text_analyzer: Any,
    csv_analyzer: Any,
    ml_analyzer: Any,
) -> tuple[list[TextContent], Any, Any] | None:
    """
    Handle data analysis tool calls.
    
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
    if name == "compute_correlation":
        if not csv_analyzer:
            return no_csv_analyzer_response(), corpus, ml_analyzer
        
        try:
            columns_str = arguments.get("columns")
            columns = [c.strip() for c in columns_str.split(",")] if columns_str else None
            method = arguments.get("method", "pearson")
            threshold = arguments.get("threshold", 0.0)
            
            corr_matrix = csv_analyzer.compute_correlation(
                columns=columns,
                threshold=threshold,
                method=method
            )
            
            if corr_matrix.empty:
                return success_response("No correlations found (possibly no numeric columns or all below threshold)"), corpus, ml_analyzer
            
            # Format the correlation matrix nicely
            response_text = f"Correlation Matrix ({method} method, threshold={threshold}):\n\n"
            response_text += corr_matrix.to_string()
            
            return success_response(response_text), corpus, ml_analyzer
            
        except Exception as e:
            logger.error(f"Error computing correlation: {e}")
            return error_response(f"Error computing correlation: {str(e)}"), corpus, ml_analyzer

    elif name == "find_significant_correlations":
        if not csv_analyzer:
            return no_csv_analyzer_response(), corpus, ml_analyzer
        
        try:
            columns_str = arguments.get("columns")
            columns = [c.strip() for c in columns_str.split(",")] if columns_str else None
            threshold = arguments.get("threshold", 0.5)
            method = arguments.get("method", "pearson")
            
            result_df = csv_analyzer.find_significant_correlations(
                columns=columns,
                threshold=threshold,
                method=method
            )
            
            if result_df.empty:
                return success_response(f"No significant correlations found above threshold {threshold}"), corpus, ml_analyzer
            
            response_text = f"Significant Correlations (|r| >= {threshold}, {method} method):\n\n"
            response_text += f"Found {len(result_df)} significant correlation(s):\n\n"
            
            for _, row in result_df.iterrows():
                response_text += f"  {row['Variable 1']} ↔ {row['Variable 2']}: {row['Correlation']:.3f}\n"
            
            return success_response(response_text), corpus, ml_analyzer
            
        except Exception as e:
            logger.error(f"Error finding correlations: {e}")
            return error_response(f"Error finding correlations: {str(e)}"), corpus, ml_analyzer

    elif name == "execute_query":
        if not csv_analyzer:
            return no_csv_analyzer_response(), corpus, ml_analyzer
        
        try:
            query = arguments.get("query")
            save_result = arguments.get("save_result", False)
            
            if not query:
                return error_response("Query parameter is required"), corpus, ml_analyzer
            
            result_df = csv_analyzer.execute_query(query, save_result=save_result)
            
            response_text = f"Query executed successfully: {query}\n\n"
            response_text += f"Result shape: {result_df.shape[0]} rows × {result_df.shape[1]} columns\n\n"
            
            if len(result_df) > 0:
                response_text += "First few rows:\n"
                response_text += result_df.head(10).to_string()
            else:
                response_text += "Query returned no rows"
            
            if save_result:
                response_text += "\n\nResult saved to DataFrame"
            
            return success_response(response_text), corpus, ml_analyzer
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return error_response(f"Error executing query: {str(e)}"), corpus, ml_analyzer

    elif name == "get_column_statistics":
        if not csv_analyzer:
            return no_csv_analyzer_response(), corpus, ml_analyzer
        
        try:
            columns_str = arguments.get("columns")
            
            if columns_str:
                columns = [c.strip() for c in columns_str.split(",")]
                stats_df = csv_analyzer.df[columns].describe()
            else:
                stats_df = csv_analyzer.df.describe()
            
            response_text = "Column Statistics:\n\n"
            response_text += stats_df.to_string()
            
            return success_response(response_text), corpus, ml_analyzer
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return error_response(f"Error getting statistics: {str(e)}"), corpus, ml_analyzer

    elif name == "get_unique_values_summary":
        if not csv_analyzer:
            return no_csv_analyzer_response(), corpus, ml_analyzer
        
        try:
            columns_str = arguments.get("columns")
            top_n = arguments.get("top_n", 10)
            
            if columns_str:
                columns = [c.strip() for c in columns_str.split(",")]
            else:
                columns = csv_analyzer.df.columns.tolist()
            
            response_text = "Unique Values Summary:\n\n"
            
            for col in columns:
                if col not in csv_analyzer.df.columns:
                    response_text += f"Column '{col}' not found\n"
                    continue
                
                unique_count = csv_analyzer.df[col].nunique()
                total_count = len(csv_analyzer.df[col])
                unique_pct = (unique_count / total_count * 100) if total_count > 0 else 0
                
                response_text += f"\n{col}:\n"
                response_text += f"  Total values: {total_count}\n"
                response_text += f"  Unique values: {unique_count} ({unique_pct:.1f}%)\n"
                
                if unique_count > 0 and unique_count <= 100:  # Only show value counts for reasonable cardinality
                    value_counts = csv_analyzer.df[col].value_counts().head(top_n)
                    response_text += f"  Top {min(top_n, len(value_counts))} values:\n"
                    for val, count in value_counts.items():
                        pct = (count / total_count * 100)
                        response_text += f"    {val}: {count} ({pct:.1f}%)\n"
                elif unique_count > 100:
                    response_text += f"  High cardinality column (>100 unique values)\n"
            
            return success_response(response_text), corpus, ml_analyzer
            
        except Exception as e:
            logger.error(f"Error getting unique values: {e}")
            return error_response(f"Error getting unique values summary: {str(e)}"), corpus, ml_analyzer

    return None
