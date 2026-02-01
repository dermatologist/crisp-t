"""
Topological Data Analysis Tools for MCP Server

This module contains tools for topological data analysis
including TDABM (Topological Data Analysis Ball Mapper).
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


def get_topological_analysis_tools() -> list[Tool]:
    """Return list of Topological Data Analysis tool definitions."""
    return [
        Tool(
            name="tdabm_analysis",
            description="""
            Perform Topological Data Analysis Ball Mapper (TDABM) to discover hidden patterns in multidimensional data. Creates point cloud revealing topological structure and variable relationships. Essential for: Exploratory multidimensional analysis (many variables simultaneously), Discovering non-linear patterns, Model-free data understanding (no assumptions about relationships).

            TDABM algorithm: Creates balls (overlapping regions) covering data points, reveals topological connectivity and clusters. Complements statistical methods (which assume distributions) and ML (which needs outcome labels).

            Parameters:
            - y_variable: Target/outcome continuous variable
            - x_variables: Predictor variables comma-separated (should be numeric/ordinal)
            - radius: Ball coverage size (default: 0.3, smaller=more detail, larger=smoother)
              - 0.1-0.2: Fine granularity, more balls, detailed patterns
              - 0.3-0.5: Balanced, typical use
              - 0.5+: Coarse aggregation, fewer balls

            Workflow: get_df_columns → retain_numeric_columns_only → tdabm_analysis(y_variable='outcome', x_variables='factor1,factor2,factor3', radius=0.3) → explore results.

            Use when: Decision trees/regression don't capture patterns, need global structure understanding, data is high-dimensional.
            Tip: Start with radius=0.3; adjust if too sparse (increase) or too dense (decrease).
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "y_variable": {
                        "type": "string",
                        "description": "Target/outcome continuous variable name",
                    },
                    "x_variables": {
                        "type": "string",
                        "description": "Comma-separated predictor variables (numeric/ordinal)",
                    },
                    "radius": {
                        "type": "number",
                        "description": "Ball coverage radius (default: 0.3, try 0.1-0.5 for detail adjustment)",
                        "default": 0.3,
                    },
                },
                "required": ["y_variable", "x_variables"],
            },
        )
    ]


def handle_topological_analysis_tool(
    name: str,
    arguments: dict[str, Any],
    corpus: Any,
    text_analyzer: Any,
    csv_analyzer: Any,
    ml_analyzer: Any,
) -> tuple[list[TextContent], Any, Any] | None:
    """Handle Topological Data Analysis tool calls.
    
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
    if name == "tdabm_analysis":
        if not corpus:
            return no_corpus_response(), corpus, ml_analyzer

        try:
            from ...tdabm import Tdabm

            y_variable = arguments.get("y_variable")
            x_variables = arguments.get("x_variables")
            radius = arguments.get("radius", 0.3)

            if not y_variable or not x_variables:
                return error_response(
                    "Both y_variable and x_variables are required"
                ), corpus, ml_analyzer

            tdabm_analyzer = Tdabm(corpus)
            result = tdabm_analyzer.generate_tdabm(
                y=y_variable, x_variables=x_variables, radius=radius, mcp=True
            )

            return success_response(
                f"TDABM Analysis Complete\n\n{result}\n\n"
                "Hint: Results are stored in corpus metadata['tdabm']\n"
                "Hint: Use save_corpus to persist the results\n"
                "Hint: Visualize with draw_tdabm or use vizcli --tdabm"
            ), corpus, ml_analyzer

        except ValueError as e:
            return error_response(
                f"Validation Error: {e}\n\n"
                "Tips:\n"
                "- Ensure corpus has a DataFrame\n"
                "- Y variable must be continuous (not binary)\n"
                "- X variables must be numeric/ordinal\n"
                "- All variables must exist in the DataFrame"
            ), corpus, ml_analyzer
        except Exception as e:
            return error_response(
                f"Error during TDABM analysis: {e}"
            ), corpus, ml_analyzer

    # Tool not handled by this module
    return None
