"""
MCP Server for CRISP-T

This module provides an MCP (Model Context Protocol) server that exposes
CRISP-T's text analysis, ML analysis, and corpus manipulation capabilities
as tools, resources, and prompts.
"""

import json
import logging
from typing import Any, cast

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    GetPromptResult,
    Prompt,
    PromptMessage,
    Resource,
    TextContent,
    Tool,
)

from ..helpers.analyzer import get_csv_analyzer, get_text_analyzer
from ..helpers.initializer import initialize_corpus
from .prompts import ANALYSIS_WORKFLOW, TRIANGULATION_GUIDE
from .tools import get_all_tools, handle_tool_call
from .utils.responses import error_response

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import ML if available
try:
    from ..ml import ML

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("ML dependencies not available")
    # Provide a placeholder for ML to satisfy type checkers when unavailable
    ML = cast(Any, None)

# Global state for the server
_corpus = None
_text_analyzer = None
_csv_analyzer = None
_ml_analyzer = None


def _init_corpus(
    inp: str | None = None,
    source: str | None = None,
    text_columns: str = "",
    ignore_words: str = "",
):
    """Initialize corpus from input path or source."""
    global _corpus, _text_analyzer, _csv_analyzer

    try:
        _corpus = initialize_corpus(
            source=source,
            inp=inp,
            comma_separated_text_columns=text_columns,
            comma_separated_ignore_words=ignore_words or "",
        )

        if _corpus:
            _text_analyzer = get_text_analyzer(_corpus, filters=[])

            # Initialize CSV analyzer if DataFrame is present
            if getattr(_corpus, "df", None) is not None:
                _csv_analyzer = get_csv_analyzer(
                    _corpus,
                    comma_separated_unstructured_text_columns=text_columns,
                    comma_separated_ignore_columns="",
                    filters=[],
                )

        return True
    except Exception as e:
        logger.exception(f"Failed to initialize corpus: {e}")
        return False


# Create the MCP server instance
app = Server("crisp-t")


@app.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources - corpus documents."""
    resources = []

    if _corpus and _corpus.documents:
        for doc in _corpus.documents:
            resources.append(
                Resource(
                    uri=cast(Any, f"corpus://document/{doc.id}"),
                    name=f"Document: {doc.name or doc.id}",
                    description=doc.description or f"Text content of document {doc.id}",
                    mimeType="text/plain",
                )
            )

    return resources


@app.read_resource()
async def read_resource(uri: Any) -> list[TextContent]:
    """Read a corpus document by URI.

    Returns a list of TextContent items to conform to MCP's expected
    function output schema for resource reads.
    """
    uri_str = str(uri)
    if not uri_str.startswith("corpus://document/"):
        raise ValueError(f"Unknown resource URI: {uri}")

    doc_id = uri_str.replace("corpus://document/", "")

    if not _corpus:
        raise ValueError("No corpus loaded. Use load_corpus tool first.")

    doc = _corpus.get_document_by_id(doc_id)
    if not doc:
        raise ValueError(f"Document not found: {doc_id}")

    return [TextContent(type="text", text=doc.text)]


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools from all tool modules."""
    return get_all_tools()




@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls by routing to appropriate tool modules."""
    global _corpus, _text_analyzer, _csv_analyzer, _ml_analyzer

    try:
        # Route to tool handlers
        result = handle_tool_call(
            name, arguments, _corpus, _text_analyzer, _csv_analyzer, _ml_analyzer
        )
        
        if result is not None:
            # Unpack the result - it's a 5-tuple
            response, _corpus, _text_analyzer, _csv_analyzer, _ml_analyzer = result
            return response
        else:
            return error_response(f"Unknown tool: {name}")

    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}", exc_info=True)
        return error_response(str(e))


@app.list_prompts()
async def list_prompts() -> list[Prompt]:
    """List available prompts."""
    return [
        Prompt(
            name="analysis_workflow",
            description="Step-by-step guide for conducting a complete CRISP-T analysis based on INSTRUCTIONS.md",
            arguments=[],
        ),
        Prompt(
            name="triangulation_guide",
            description="Guide for triangulating qualitative and quantitative findings",
            arguments=[],
        ),
    ]


@app.get_prompt()
async def get_prompt(
    name: str, arguments: dict[str, str] | None = None
) -> GetPromptResult:
    """Get a specific prompt."""

    if name == "analysis_workflow":
        return GetPromptResult(
            description="Complete analysis workflow for CRISP-T",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=ANALYSIS_WORKFLOW,
                    ),
                )
            ],
        )

    elif name == "triangulation_guide":
        return GetPromptResult(
            description="Guide for triangulating qualitative and quantitative findings",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=TRIANGULATION_GUIDE,
                    ),
                )
            ],
        )

    raise ValueError(f"Unknown prompt: {name}")


async def main():
    """Main entry point for the MCP server."""
    # Print startup message to stderr so it doesn't interfere with MCP protocol
    import sys

    print("=" * 60, file=sys.stderr)
    print("🚀 CRISP-T MCP Server Starting...", file=sys.stderr)
    print(
        "   Model Context Protocol (MCP) Server for Qualitative Research",
        file=sys.stderr,
    )
    print("   Ready to accept connections from MCP clients", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())
