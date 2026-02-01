"""
Corpus Management Tools for MCP Server

This module contains tools for managing corpus data including loading, saving,
document operations, and relationship management.
"""

import json
import logging
from typing import Any

from mcp.types import TextContent, Tool

from ...helpers.initializer import initialize_corpus
from ...read_data import ReadData
from ..utils.responses import (
    error_response,
    no_corpus_response,
    success_response,
)

logger = logging.getLogger(__name__)


def get_corpus_management_tools() -> list[Tool]:
    """Return list of corpus management tool definitions."""
    return [
        Tool(
            name="load_corpus",
            description="Load a corpus from a folder containing corpus.json or from a source directory/URL. Essential starting point for all analysis workflows. If using 'source', CRISP-T will import text files (.txt, .pdf) and CSV data into a new corpus structure.\n\t    Workflow tips:\n\t    - If corpus already exists: use 'inp' parameter\n\t    - If importing raw data: use 'source' parameter to auto-create corpus\n\t    - For CSV with text columns: specify 'text_columns' to mark free-text fields\n\t    - Use 'ignore_words' to exclude stop words during initial processing\n\t    - Always call this first before any analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "inp": {
                        "type": "string",
                        "description": "Path to folder containing corpus.json",
                    },
                    "source": {
                        "type": "string",
                        "description": "Source directory or URL to read data from",
                    },
                    "text_columns": {
                        "type": "string",
                        "description": "Comma-separated text column names (for CSV data)",
                    },
                    "ignore_words": {
                        "type": "string",
                        "description": "Comma-separated words to ignore during analysis",
                    },
                },
            },
        ),
        Tool(
            name="save_corpus",
            description="Save the current corpus to a folder as corpus.json and corpus_df.csv (if DataFrame exists). Persists all work including documents, metadata, relationships, and analysis results.\n\t    Use cases: After document modifications, text analysis, linking documents to numeric data, filtering/transforming corpus, or creating analysis checkpoints.\n\t    Tip: Save frequently to preserve work. Different output folders create separate analysis branches.",
            inputSchema={
                "type": "object",
                "properties": {
                    "out": {
                        "type": "string",
                        "description": "Output folder path to save corpus",
                    }
                },
                "required": ["out"],
            },
        ),
        Tool(
            name="add_document",
            description="Add a new document to the corpus. Expands your dataset with new text entries or integrates external documents into existing corpus.\n\t    Use when: Incrementally building corpus, manually adding external documents, correcting/updating specific documents, or combining multiple corpora.\n\t    Tips: doc_id must be unique within corpus. name field is optional but recommended. Consider adding before analysis for best results.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "Unique document ID"},
                    "text": {"type": "string", "description": "Document text content"},
                    "name": {"type": "string", "description": "Optional document name"},
                },
                "required": ["doc_id", "text"],
            },
        ),
        Tool(
            name="remove_document",
            description="Remove a document from the corpus by ID. Useful for data cleaning, curation, or excluding outliers/irrelevant entries.\n\t    Use for: Removing duplicates, excluding irrelevant/off-topic documents, correcting data quality issues, or curating corpus to specific criteria.\n\t    Tip: Call list_documents first to identify document IDs to remove.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "Document ID to remove"}
                },
                "required": ["doc_id"],
            },
        ),
        Tool(
            name="get_document",
            description="Retrieve a specific document by ID to inspect its full text and metadata. Essential for validating data, reviewing specific cases, or extracting quotes.\n\t    Use for: Validating document content, extracting passages for reporting, reviewing documents with specific characteristics, or debugging metadata/assignments.\n\t    Tip: Use with assign_topics or get_relationships to understand document context.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "Document ID"}
                },
                "required": ["doc_id"],
            },
        ),
        Tool(
            name="list_documents",
            description="List all document IDs in the corpus. Returns IDs for all available documents, useful for exploration, validation, and batch operations.\n\t    Use to: Explore corpus structure/size, identify specific documents for further analysis, validate document import/creation, or generate reference lists for reporting.\n\t    Workflow: Call this after load_corpus to verify data was loaded correctly.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="add_relationship",
            description="Establish explicit links between text findings and numeric variables for mixed-methods triangulation. Documents connections discovered through analysis.\n\t    Format: first|second|relation where first='text:keyword', second='num:column', relation='predicts'/'correlates'/'contrasts'.\n\t    Workflow: Add relationships after running assign_topics, topic_modeling, or sentiment_analysis to link findings to outcomes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "first": {
                        "type": "string",
                        "description": "First entity (e.g., 'text:keyword')",
                    },
                    "second": {
                        "type": "string",
                        "description": "Second entity (e.g., 'numb:column')",
                    },
                    "relation": {
                        "type": "string",
                        "description": "Relationship type (e.g., 'correlates')",
                    },
                },
                "required": ["first", "second", "relation"],
            },
        ),
        Tool(
            name="get_relationships",
            description="Retrieve all established relationships between text and numeric data. Essential for understanding corpus-wide connections and triangulation findings.\n\t    Use to: Review all documented connections, validate relationship patterns, export findings for reporting, or plan further analysis based on known patterns.\n\t    Returns: List of all text↔numeric relationships with their types.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_relationships_for_keyword",
            description="Find all relationships connected to a specific keyword, topic, or theme. Useful for exploring how particular concepts relate to numeric outcomes.\n\t    Examples: Find all metrics related to 'satisfaction' keyword, explore connections for specific topics, trace theme relationships through data.\n\t    Tip: Use after add_relationship to verify links were created correctly.",
            inputSchema={
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Keyword to search for",
                    }
                },
                "required": ["keyword"],
            },
        ),
    ]


def _init_corpus(
    corpus: Any,
    text_analyzer: Any,
    csv_analyzer: Any,
    inp: str | None = None,
    source: str | None = None,
    text_columns: str = "",
    ignore_words: str = "",
) -> tuple[Any, Any, Any, bool]:
    """Initialize corpus from input path or source.
    
    Returns:
        Tuple of (corpus, text_analyzer, csv_analyzer, success)
    """
    from ...helpers.analyzer import get_csv_analyzer, get_text_analyzer
    
    try:
        corpus = initialize_corpus(
            source=source,
            inp=inp,
            comma_separated_text_columns=text_columns,
            comma_separated_ignore_words=ignore_words or "",
        )

        if corpus:
            text_analyzer = get_text_analyzer(corpus, filters=[])

            # Initialize CSV analyzer if DataFrame is present
            if getattr(corpus, "df", None) is not None:
                csv_analyzer = get_csv_analyzer(
                    corpus,
                    comma_separated_unstructured_text_columns=text_columns,
                    comma_separated_ignore_columns="",
                    filters=[],
                )

        return corpus, text_analyzer, csv_analyzer, True
    except Exception as e:
        logger.exception(f"Failed to initialize corpus: {e}")
        return corpus, text_analyzer, csv_analyzer, False


def handle_corpus_management_tool(
    name: str,
    arguments: dict[str, Any],
    corpus: Any,
    text_analyzer: Any,
    csv_analyzer: Any,
    ml_analyzer: Any,
) -> tuple[list[TextContent], Any, Any] | None:
    """Handle corpus management tool calls.
    
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
    if name == "load_corpus":
        inp = arguments.get("inp")
        source = arguments.get("source")
        text_columns = arguments.get("text_columns", "")
        ignore_words = arguments.get("ignore_words", "")

        corpus, text_analyzer, csv_analyzer, success = _init_corpus(
            corpus, text_analyzer, csv_analyzer, inp, source, text_columns, ignore_words
        )
        
        if success:
            doc_count = len(corpus.documents) if corpus else 0
            return (
                success_response(f"Corpus loaded successfully with {doc_count} document(s)"),
                corpus,
                ml_analyzer,
            )
        else:
            return error_response("Failed to load corpus"), corpus, ml_analyzer

    elif name == "save_corpus":
        if not corpus:
            return no_corpus_response(), corpus, ml_analyzer

        out = arguments["out"]
        read_data = ReadData(corpus=corpus)
        read_data.write_corpus_to_json(out, corpus=corpus)
        return success_response(f"Corpus saved to {out}"), corpus, ml_analyzer

    elif name == "add_document":
        if not corpus:
            return no_corpus_response(), corpus, ml_analyzer

        from ...model.document import Document

        doc = Document(
            id=arguments["doc_id"],
            text=arguments["text"],
            name=arguments.get("name"),
            description=None,
            score=0.0,
            metadata={},
        )
        corpus.add_document(doc)
        return success_response(f"Document {arguments['doc_id']} added"), corpus, ml_analyzer

    elif name == "remove_document":
        if not corpus:
            return no_corpus_response(), corpus, ml_analyzer

        corpus.remove_document_by_id(arguments["doc_id"])
        return success_response(f"Document {arguments['doc_id']} removed"), corpus, ml_analyzer

    elif name == "get_document":
        if not corpus:
            return no_corpus_response(), corpus, ml_analyzer

        doc = corpus.get_document_by_id(arguments["doc_id"])
        if doc:
            return (
                success_response(json.dumps(doc.model_dump(), indent=2, default=str)),
                corpus,
                ml_analyzer,
            )
        return error_response("Document not found"), corpus, ml_analyzer

    elif name == "list_documents":
        if not corpus:
            return no_corpus_response(), corpus, ml_analyzer

        doc_ids = corpus.get_all_document_ids()
        return success_response(json.dumps(doc_ids, indent=2)), corpus, ml_analyzer

    elif name == "add_relationship":
        if not corpus:
            return no_corpus_response(), corpus, ml_analyzer

        corpus.add_relationship(
            arguments["first"], arguments["second"], arguments["relation"]
        )
        return success_response("Relationship added"), corpus, ml_analyzer

    elif name == "get_relationships":
        if not corpus:
            return no_corpus_response(), corpus, ml_analyzer

        rels = corpus.get_relationships()
        return success_response(json.dumps(rels, indent=2)), corpus, ml_analyzer

    elif name == "get_relationships_for_keyword":
        if not corpus:
            return [TextContent(type="text", text="No corpus loaded")], corpus, ml_analyzer

        rels = corpus.get_all_relationships_for_keyword(arguments["keyword"])
        return [TextContent(type="text", text=json.dumps(rels, indent=2))], corpus, ml_analyzer

    # Tool not handled by this module
    return None
