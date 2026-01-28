"""Response utilities for MCP server handlers."""
from mcp.types import TextContent


def success_response(text: str) -> list[TextContent]:
    """Create a success response with the given text."""
    return [TextContent(type="text", text=text)]


def error_response(message: str) -> list[TextContent]:
    """Create an error response with the given message."""
    return [TextContent(type="text", text=f"Error: {message}")]


def no_corpus_response() -> list[TextContent]:
    """Standard response when corpus is not loaded."""
    return error_response("No corpus loaded. Please use 'load_corpus' first.")


def no_csv_analyzer_response() -> list[TextContent]:
    """Standard response when CSV analyzer is not available."""
    return error_response("No CSV data loaded. Load corpus with numeric data first.")


def no_text_analyzer_response() -> list[TextContent]:
    """Standard response when text analyzer is not available."""
    return error_response("No text data loaded. Load corpus with text documents first.")
