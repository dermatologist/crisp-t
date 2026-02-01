"""
Temporal Analysis Tools for MCP Server

This module contains tools for temporal analysis of text and numeric data over time.
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


def get_temporal_analysis_tools() -> list[Tool]:
    """Return list of temporal analysis tool definitions."""
    return [
        Tool(
            name="temporal_link_by_time",
            description="""
            Link documents to dataframe rows based on timestamps to create text↔numeric relationships over time. Essential for: Studying how text themes evolve (sentiment over time), Validating numeric trends with qualitative context (what happened during spike?), Mixed-methods temporal analysis.

            Three linking methods:
            - 'nearest': Document links to closest timestamp (best for: sparse data, one event per period)
            - 'window': Document links to all rows within time window (best for: capturing events in time range, default window=300 seconds)
            - 'sequence': Documents grouped by period then linked (best for: regular periods like D/W/M/Y)

            Parameters:
            - method: One of nearest, window, sequence
            - time_column: DataFrame column with timestamps (default: 'timestamp')
            - window_seconds: Time range in seconds for 'window' (default: 300=5 min)
            - period: For 'sequence': 'D' (day), 'W' (week), 'M' (month), 'Y' (year) - affects grouping granularity

            Workflow: Ensure documents have doc.time field → temporal_link_by_time(method='sequence', period='W') → get_relationships to validate links → Analyze text+numeric patterns over time.
            Tip: Use temporal_sentiment_trend after linking to see sentiment evolution.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "Linking method: 'nearest' (closest time), 'window' (within time range), 'sequence' (grouped periods)",
                        "enum": ["nearest", "window", "sequence"],
                    },
                    "time_column": {
                        "type": "string",
                        "description": "DataFrame timestamp column (default: 'timestamp')",
                        "default": "timestamp",
                    },
                    "window_seconds": {
                        "type": "number",
                        "description": "Time window in seconds for 'window' method (default: 300=5min, try 60-3600)",
                        "default": 300,
                    },
                    "period": {
                        "type": "string",
                        "description": "Period for 'sequence': 'D'(day), 'W'(week), 'M'(month), 'Y'(year)",
                        "default": "W",
                    },
                },
                "required": ["method"],
            },
        ),
        Tool(
            name="temporal_filter",
            description="""
            Filter corpus by time range (ISO 8601 format). Removes documents/rows outside range. Essential for: Studying specific time periods, Validating temporal patterns, Reducing scope for focused analysis.

            Parameters:
            - start_time, end_time: ISO format '2025-01-01T00:00:00'
            - time_column: DataFrame timestamp column (default: 'timestamp')

            Workflow: temporal_summary (initial exploration) → identify interesting period → temporal_filter(start_time, end_time) → analyze subset → temporal_sentiment_trend to see patterns in period.

            Returns: Filtered corpus with documents/rows only in time range. Use temporal_sentiment_trend or temporal_topics after filtering.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "description": "Start time ISO 8601 (e.g., '2025-01-01T00:00:00')",
                    },
                    "end_time": {
                        "type": "string",
                        "description": "End time ISO 8601 (e.g., '2025-12-31T23:59:59')",
                    },
                    "time_column": {
                        "type": "string",
                        "description": "DataFrame timestamp column (default: 'timestamp')",
                        "default": "timestamp",
                    },
                },
            },
        ),
        Tool(
            name="temporal_summary",
            description="""
            Generate temporal summary showing aggregated statistics and document counts per time period. Essential for: Exploratory temporal analysis (what happened when?), Identifying interesting periods for deeper analysis, Validating temporal patterns in data.

            Parameters:
            - period: 'D' (day), 'W' (week), 'M' (month), 'Y' (year) - choose granularity matching your data
              - Daily (D): For fine-grained events, real-time data
              - Weekly (W): Common for interview/survey data
              - Monthly (M): For long-term trends
              - Yearly (Y): For multi-year studies
            - time_column: DataFrame timestamp column (default: 'timestamp')

            Output: Counts and statistics per period (e.g., document count, numeric means).

            Workflow: temporal_summary(period='W') to see overall pattern → temporal_filter to zoom into interesting period → temporal_sentiment_trend to analyze mood.
            Tip: If too sparse, try longer period (M instead of W). If too aggregated, try shorter (W instead of M).
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "description": "Time period: 'D'(day), 'W'(week-default), 'M'(month), 'Y'(year)",
                        "default": "W",
                    },
                    "time_column": {
                        "type": "string",
                        "description": "Timestamp column in dataframe",
                        "default": "timestamp",
                    },
                },
            },
        ),
        Tool(
            name="temporal_sentiment_trend",
            description="""
            Analyze sentiment trends over time. Requires documents to have sentiment metadata.
            Returns aggregated sentiment scores per time period.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "description": "Time period: 'D' (day), 'W' (week), 'M' (month)",
                        "default": "W",
                    },
                    "aggregation": {
                        "type": "string",
                        "description": "Aggregation method: 'mean', 'median', 'max', 'min'",
                        "default": "mean",
                    },
                },
            },
        ),
        Tool(
            name="temporal_topics",
            description="""
            Extract topics over time periods. Shows how topics evolve and change over time.
            Works best with documents that have topic metadata from topic modeling.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "description": "Time period: 'D', 'W', 'M'",
                        "default": "W",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top topics per period",
                        "default": 5,
                    },
                },
            },
        ),
    ]


def handle_temporal_analysis_tool(
    name: str,
    arguments: dict[str, Any],
    corpus: Any,
    text_analyzer: Any,
    csv_analyzer: Any,
    ml_analyzer: Any,
) -> tuple[list[TextContent], Any, Any] | None:
    """Handle temporal analysis tool calls.
    
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
    if name == "temporal_link_by_time":
        if not corpus:
            return no_corpus_response(), corpus, ml_analyzer

        try:
            from datetime import timedelta

            from ...temporal import TemporalAnalyzer

            method = arguments.get("method", "nearest")
            time_column = arguments.get("time_column", "timestamp")
            analyzer = TemporalAnalyzer(corpus)

            if method == "nearest":
                corpus = analyzer.link_by_nearest_time(time_column=time_column)
                return (
                    success_response(
                        "Documents linked to nearest dataframe rows by time"
                    ),
                    corpus,
                    ml_analyzer,
                )

            elif method == "window":
                window_seconds = arguments.get("window_seconds", 300)
                window = timedelta(seconds=window_seconds)
                corpus = analyzer.link_by_time_window(
                    time_column=time_column,
                    window_before=window,
                    window_after=window,
                )
                return (
                    success_response(
                        f"Documents linked within ±{window_seconds}s time window"
                    ),
                    corpus,
                    ml_analyzer,
                )

            elif method == "sequence":
                period = arguments.get("period", "W")
                corpus = analyzer.link_by_sequence(
                    time_column=time_column, period=period
                )
                return (
                    success_response(f"Documents linked in {period} sequences"),
                    corpus,
                    ml_analyzer,
                )

            else:
                return (
                    error_response(f"Unknown method: {method}"),
                    corpus,
                    ml_analyzer,
                )

        except Exception as e:
            return (
                error_response(f"Error in temporal linking: {e}"),
                corpus,
                ml_analyzer,
            )

    elif name == "temporal_filter":
        if not corpus:
            return no_corpus_response(), corpus, ml_analyzer

        try:
            from ...temporal import TemporalAnalyzer

            start_time = arguments.get("start_time")
            end_time = arguments.get("end_time")
            time_column = arguments.get("time_column", "timestamp")

            analyzer = TemporalAnalyzer(corpus)
            corpus = analyzer.filter_by_time_range(
                start_time=start_time, end_time=end_time, time_column=time_column
            )

            doc_count = len(corpus.documents)
            df_count = len(corpus.df) if corpus.df is not None else 0
            return (
                success_response(
                    f"Corpus filtered by time range: {doc_count} documents, {df_count} dataframe rows"
                ),
                corpus,
                ml_analyzer,
            )

        except Exception as e:
            return (
                error_response(f"Error in temporal filtering: {e}"),
                corpus,
                ml_analyzer,
            )

    elif name == "temporal_summary":
        if not corpus:
            return no_corpus_response(), corpus, ml_analyzer

        try:
            from ...temporal import TemporalAnalyzer

            period = arguments.get("period", "W")
            time_column = arguments.get("time_column", "timestamp")

            analyzer = TemporalAnalyzer(corpus)
            summary = analyzer.get_temporal_summary(
                time_column=time_column, period=period
            )

            if not summary.empty:
                response_text = f"Temporal Summary ({period} periods):\n\n"
                response_text += summary.to_string()
                return success_response(response_text), corpus, ml_analyzer
            else:
                return (
                    error_response("No temporal data available for summary"),
                    corpus,
                    ml_analyzer,
                )

        except Exception as e:
            return (
                error_response(f"Error in temporal summary: {e}"),
                corpus,
                ml_analyzer,
            )

    elif name == "temporal_sentiment_trend":
        if not corpus:
            return no_corpus_response(), corpus, ml_analyzer

        try:
            from ...temporal import TemporalAnalyzer

            period = arguments.get("period", "W")
            aggregation = arguments.get("aggregation", "mean")

            analyzer = TemporalAnalyzer(corpus)
            trend = analyzer.get_temporal_sentiment_trend(
                period=period, aggregation=aggregation
            )

            if not trend.empty:
                response_text = f"Temporal Sentiment Trend ({period} periods, {aggregation}):\n\n"
                response_text += trend.to_string()
                return success_response(response_text), corpus, ml_analyzer
            else:
                return (
                    error_response(
                        "No sentiment data available. Run sentiment analysis first."
                    ),
                    corpus,
                    ml_analyzer,
                )

        except Exception as e:
            return (
                error_response(f"Error in temporal sentiment: {e}"),
                corpus,
                ml_analyzer,
            )

    elif name == "temporal_topics":
        if not corpus:
            return no_corpus_response(), corpus, ml_analyzer

        try:
            from ...temporal import TemporalAnalyzer

            period = arguments.get("period", "W")
            top_n = arguments.get("top_n", 5)

            analyzer = TemporalAnalyzer(corpus)
            topics = analyzer.get_temporal_topics(period=period, top_n=top_n)

            if topics:
                response_text = (
                    f"Temporal Topics (top {top_n} per {period} period):\n\n"
                )
                for period_key, topic_list in topics.items():
                    response_text += f"{period_key}: {', '.join(topic_list)}\n"
                return success_response(response_text), corpus, ml_analyzer
            else:
                return (
                    error_response(
                        "No temporal data available for topic extraction"
                    ),
                    corpus,
                    ml_analyzer,
                )

        except Exception as e:
            return (
                error_response(f"Error in temporal topics: {e}"),
                corpus,
                ml_analyzer,
            )

    # Tool not handled by this module
    return None
