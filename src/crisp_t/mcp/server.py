"""
MCP Server for CRISP-T

This module provides an MCP (Model Context Protocol) server that exposes
CRISP-T's text analysis, ML analysis, and corpus manipulation capabilities
as tools, resources, and prompts.
"""

import json
import logging
from typing import Any, Optional, cast

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

from ..cluster import Cluster
from ..csv import Csv
from ..helpers.analyzer import get_csv_analyzer, get_text_analyzer
from ..helpers.initializer import initialize_corpus
from ..read_data import ReadData
from ..sentiment import Sentiment
from ..text import Text

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
    inp: Optional[str] = None,
    source: Optional[str] = None,
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
        logger.error(f"Failed to initialize corpus: {e}")
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
    """List available tools."""
    tools = [
        # Corpus management tools
        Tool(
            name="load_corpus",
            description="Load a corpus from a folder containing corpus.json or from a source directory/URL. Run this first before any analysis.",
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
            description="Save the current corpus to a folder as corpus.json. Use this to persist your work after analysis or transformation.",
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
            description="Add a new document to the corpus. Use this to expand your dataset with new text entries.",
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
            description="Remove a document from the corpus by ID. Use this to clean up or curate your corpus.",
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
            description="Get a document by ID from the corpus. Use this to inspect or retrieve specific documents for review.",
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
            description="List all document IDs in the corpus. Use this to enumerate all available documents for batch operations.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="add_relationship",
            description="Add a relationship between text keywords and numeric columns. Link topic modeling results with DataFrame columns for triangulation.",
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
            description="Get all relationships in the corpus. Review established links between text and numeric data.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_relationships_for_keyword",
            description="Get relationships involving a specific keyword. Explore connections for a particular topic or term.",
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
        # ! NLP/Text Analysis Tools
        # Tool(
        #     name="generate_coding_dictionary",
        #     description="""
        #     Generate a qualitative coding dictionary with categories (verbs), properties (nouns), and dimensions (adjectives/adverbs). Useful for understanding the main themes and concepts in the corpus.
        #     Tips:
        #       - Use ignore to exclude common but uninformative words.
        #       - Use filters to narrow down documents based on metadata (key=value).
        #       - Adjust num (categories) and top_n (items per section).
        #     """,
        #     inputSchema={
        #         "type": "object",
        #         "properties": {
        #             "num": {
        #                 "type": "integer",
        #                 "description": "Number of categories to extract",
        #                 "default": 3,
        #             },
        #             "top_n": {
        #                 "type": "integer",
        #                 "description": "Top N items per category",
        #                 "default": 3,
        #             },
        #             "ignore": {
        #                 "type": "array",
        #                 "description": "List of words to ignore",
        #                 "items": {"type": "string"},
        #             },
        #             "filters": {
        #                 "type": "array",
        #                 "description": "Filters to apply on documents (key=value or key:value)",
        #                 "items": {"type": "string"},
        #             },
        #         },
        #     },
        # ),
        # Tool(
        #     name="topic_modeling",
        #     description="""
        #     Perform LDA topic modeling to discover latent topics in the corpus. Returns topics with their associated keywords and weights, useful for categorizing documents by theme.
        #     Tips:
        #       - Set num_topics (number of topics).
        #       - Set num_words (words to show per topic).
        #     """,
        #     inputSchema={
        #         "type": "object",
        #         "properties": {
        #             "num_topics": {
        #                 "type": "integer",
        #                 "description": "Number of topics to generate",
        #                 "default": 3,
        #             },
        #             "num_words": {
        #                 "type": "integer",
        #                 "description": "Number of words per topic",
        #                 "default": 5,
        #             },
        #         },
        #     },
        # ),
        Tool(
            name="assign_topics",
            description="""
            Assign documents to their dominant topics, themes and keywords with contribution percentages.
            These topic assignments can be used as keywords to filter or categorize documents.

            Note: Use the results to create keywords for filtering/categorization.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "num_topics": {
                        "type": "integer",
                        "description": "Number of topics (should match topic_modeling)",
                        "default": 3,
                    }
                },
            },
        ),
        Tool(
            name="extract_categories",
            description="""
            Extract common categories/concepts from the corpus as bag-of-terms with weights

            Tip: Adjust num to change how many categories are returned.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "num": {
                        "type": "integer",
                        "description": "Number of categories",
                        "default": 10,
                    }
                },
            },
        ),
        Tool(
            name="generate_summary",
            description="""
            Generate an extractive text summary of the entire corpus

            Tip: Increase weight for longer summaries.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "weight": {
                        "type": "integer",
                        "description": "Summary weight/length parameter",
                        "default": 10,
                    }
                },
            },
        ),
        Tool(
            name="sentiment_analysis",
            description="""
            Perform VADER sentiment analysis on the corpus, providing positive, negative, neutral, and compound scores

            Tips:
              - Set documents=true to analyze at document level.
              - Set verbose=true for detailed output.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "boolean",
                        "description": "Analyze at document level",
                        "default": False,
                    },
                    "verbose": {
                        "type": "boolean",
                        "description": "Verbose output",
                        "default": True,
                    },
                },
            },
        ),
        # Text/Corpus filtering tools
        Tool(
            name="filter_documents",
            description="Filter corpus documents based on a metadata key/value, document id, or name substring. Updates the active corpus.",
            inputSchema={
                "type": "object",
                "properties": {
                    "metadata_key": {
                        "type": "string",
                        "description": "Metadata key to search (optional if filtering by id/name)",
                    },
                    "metadata_value": {
                        "type": "string",
                        "description": "Value to look for in metadata, id, or name",
                    },
                },
                "required": ["metadata_value"],
            },
        ),
        Tool(
            name="document_count",
            description="Return the number of documents currently in the active corpus (after any filters).",
            inputSchema={"type": "object", "properties": {}},
        ),
        # DataFrame/CSV Tools
        Tool(
            name="get_df_columns",
            description="Get all column names from the DataFrame. Use this to inspect available features for analysis.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_df_row_count",
            description="Get the number of rows in the DataFrame. Useful for understanding dataset size before analysis.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_df_row",
            description="Get a specific row from the DataFrame by index. Use this to inspect individual records for debugging or exploration.",
            inputSchema={
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "description": "Row index"}
                },
                "required": ["index"],
            },
        ),
        # CSV Column/DataFrame operations
        Tool(
            name="bin_a_column",
            description="Bin a numeric column into a specified number of bins.",
            inputSchema={
                "type": "object",
                "properties": {
                    "column_name": {
                        "type": "string",
                        "description": "Name of the numeric column to bin",
                    },
                    "bins": {
                        "type": "integer",
                        "description": "Number of bins",
                        "default": 2,
                    },
                },
                "required": ["column_name"],
            },
        ),
        Tool(
            name="one_hot_encode_column",
            description="One-hot encode a specific categorical column.",
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
            description="Filter DataFrame rows where a column equals a specific value.",
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
            description="Apply random oversampling to balance classes (requires prior X/y preparation).",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="restore_oversample",
            description="Restore X and y to their original (pre-oversampling) values.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_column_types",
            description="Get data types of all DataFrame columns.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_column_values",
            description="Get values from a specific DataFrame column.",
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
            description="Retain only numeric columns in the DataFrame.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="reset_corpus_state",
            description="Reset the global corpus, text analyzer, and CSV analyzer state. Clear all loaded data and start fresh.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]

    # Add ML tools if available
    if ML_AVAILABLE:
        tools.extend(
            [
                Tool(
                    name="kmeans_clustering",
                    description="""
                Perform K-Means clustering on numeric data. Useful for segmenting data into groups based on similarity.
                Required: specify columns to include as a comma-separated list (include).

                Args:
                    num_clusters (int): The number of clusters to form.
                    include (str): Comma-separated columns to include in clustering.
                """,
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "num_clusters": {
                                "type": "integer",
                                "description": "Number of clusters",
                                "default": 3,
                            },
                            "outcome": {
                                "type": "string",
                                "description": "Optional outcome variable to exclude",
                            },
                            "include": {
                                "type": "string",
                                "description": "Comma-separated list of columns to include",
                            },
                        },
                        "required": ["include"],
                    },
                ),
                Tool(
                    name="decision_tree_classification",
                    description="""
                Train a decision tree classifier and return variable importance rankings. Shows which features are most predictive of the outcome.
                Required: specify columns to include in the classification as a comma-separated list (include).

                    Args:
                        outcome (str): The target variable for classification.
                        top_n (int): The number of top features to return.
                        include (str): Comma-separated list of columns to include.
                """,
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "outcome": {
                                "type": "string",
                                "description": "Target/outcome variable",
                            },
                            "top_n": {
                                "type": "integer",
                                "description": "Top N important features",
                                "default": 10,
                            },
                            "include": {
                                "type": "string",
                                "description": "Comma-separated columns to include",
                            },
                        },
                        "required": ["outcome", "include"],
                    },
                ),
                Tool(
                    name="svm_classification",
                    description="""
                Perform SVM classification and return confusion matrix
                Required: specify columns to include in the classification as a comma-separated list (include).

                Args:
                    outcome (str): The target variable for classification.
                    include (str): Comma-separated list of columns to include.
                """,
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "outcome": {
                                "type": "string",
                                "description": "Target/outcome variable",
                            },
                            "include": {
                                "type": "string",
                                "description": "Comma-separated columns to include",
                            },
                        },
                        "required": ["outcome", "include"],
                    },
                ),
                Tool(
                    name="neural_network_classification",
                    description="""
                Train a neural network classifier and return predictions with accuracy
                Required: specify columns to include in the classification as a comma-separated list (include).

                Args:
                    outcome (str): The target variable for classification.
                    include (str): Comma-separated list of columns to include.
                """,
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "outcome": {
                                "type": "string",
                                "description": "Target/outcome variable",
                            },
                            "include": {
                                "type": "string",
                                "description": "Comma-separated columns to include",
                            },
                        },
                        "required": ["outcome", "include"],
                    },
                ),
                Tool(
                    name="regression_analysis",
                    description="""
                Perform linear or logistic regression (auto-detects based on outcome). Returns coefficients for each factor/predictor, showing their relationship with the outcome variable.
                Required: specify columns to include in the regression as a comma-separated list (include).

                Args:
                    outcome (str): The target variable for regression.
                    include (str): Comma-separated list of columns to include.
                """,
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "outcome": {
                                "type": "string",
                                "description": "Target/outcome variable",
                            },
                            "include": {
                                "type": "string",
                                "description": "Comma-separated columns to include",
                            },
                        },
                        "required": ["outcome", "include"],
                    },
                ),
                Tool(
                    name="pca_analysis",
                    description="""
                Perform Principal Component Analysis for dimensionality reduction
                Required: specify columns to include in the PCA as a comma-separated list (include).

                Args:
                    outcome (str): The variable to exclude from PCA.
                    n_components (int): The number of components to keep.
                    include (str): Comma-separated list of columns to include.

                """,
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "outcome": {
                                "type": "string",
                                "description": "Variable to exclude from PCA",
                            },
                            "n_components": {
                                "type": "integer",
                                "description": "Number of components",
                                "default": 3,
                            },
                            "include": {
                                "type": "string",
                                "description": "Comma-separated columns to include",
                            },
                        },
                        "required": ["outcome", "include"],
                    },
                ),
                Tool(
                    name="association_rules",
                    description="""
                Generate association rules using Apriori algorithm
                Required: specify columns to include in the analysis as a comma-separated list (include).

                Args:
                    outcome (str): Variable to exclude from rules mining.
                    min_support (int): Minimum support as percent (1-99).
                    min_threshold (int): Minimum confidence as percent (1-99).
                    include (str): Comma-separated list of columns to include.
                """,
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "outcome": {
                                "type": "string",
                                "description": "Variable to exclude",
                            },
                            "min_support": {
                                "type": "integer",
                                "description": "Min support (1-99)",
                                "default": 50,
                            },
                            "min_threshold": {
                                "type": "integer",
                                "description": "Min threshold (1-99)",
                                "default": 50,
                            },
                            "include": {
                                "type": "string",
                                "description": "Comma-separated columns to include",
                            },
                        },
                        "required": ["outcome", "include"],
                    },
                ),
                Tool(
                    name="knn_search",
                    description="""
                Find K-nearest neighbors for a specific record
                Required: specify columns to include in the search as a comma-separated list (include).

                Args:
                    outcome (str): The target variable (excluded from features).
                    n (int): The number of neighbors to find.
                    record (int): The record index (1-based) to find neighbors for.
                    include (str): Comma-separated columns to include.
                """,
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "outcome": {
                                "type": "string",
                                "description": "Target variable",
                            },
                            "n": {
                                "type": "integer",
                                "description": "Number of neighbors",
                                "default": 3,
                            },
                            "record": {
                                "type": "integer",
                                "description": "Record index (1-based)",
                                "default": 1,
                            },
                            "include": {
                                "type": "string",
                                "description": "Comma-separated columns to include",
                            },
                        },
                        "required": ["outcome", "include"],
                    },
                ),
                Tool(
                    name="lstm_text_classification",
                    description="""
                Train an LSTM (Long Short-Term Memory) model on text documents to predict an outcome variable.
                This tool can be used to see if the texts converge towards predicting the outcome.
                
                Requirements:
                    - Text documents must be loaded in the corpus
                    - An 'id' column must exist in the DataFrame to align documents with outcomes
                    - The outcome variable must be binary (two classes)
                
                Args:
                    outcome (str): The target variable to predict (must be binary).
                
                Note: This tool tests convergence between textual content and numeric outcomes.
                """,
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "outcome": {
                                "type": "string",
                                "description": "Binary target variable to predict",
                            },
                        },
                        "required": ["outcome"],
                    },
                ),
            ]
        )

    # Semantic search tools
    tools.extend(
        [
            Tool(
                name="semantic_search",
                description="Perform semantic search to find documents similar to a query using ChromaDB. Returns documents based on semantic similarity.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query text",
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Number of similar documents to return (default: 5)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="find_similar_documents",
                description="Find documents similar to a given set of reference documents based on semantic similarity. This tool is particularly useful for literature reviews and qualitative research where you want to find additional documents that are similar to a set of known relevant documents. It can also be used to identify documents with similar themes, topics, or content for grouping and analysis purposes.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "document_ids": {
                            "type": "string",
                            "description": "A single document ID or comma-separated list of document IDs to use as reference",
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Number of similar documents to return (default: 5)",
                            "default": 5,
                        },
                        "threshold": {
                            "type": "number",
                            "description": "Minimum similarity threshold (0-1). Only documents with similarity above this value are returned (default: 0.7)",
                            "default": 0.7,
                        },
                    },
                    "required": ["document_ids"],
                },
            ),
            Tool(
                name="semantic_chunk_search",
                description="Perform semantic search on chunks of a specific document to find relevant sections. This tool is useful for coding/annotating documents by identifying chunks that match specific concepts or themes. Returns matching text chunks that can be used for qualitative analysis or document annotation.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query text (concept or set of concepts to search for)",
                        },
                        "doc_id": {
                            "type": "string",
                            "description": "Document ID to search within",
                        },
                        "threshold": {
                            "type": "number",
                            "description": "Minimum similarity threshold (0-1). Only chunks with similarity above this value are returned (default: 0.5)",
                            "default": 0.5,
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Maximum number of chunks to retrieve before filtering (default: 10)",
                            "default": 10,
                        },
                    },
                    "required": ["query", "doc_id"],
                },
            ),
            Tool(
                name="export_metadata_df",
                description="Export ChromaDB collection metadata as a pandas DataFrame. Useful for further analysis of document metadata.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "metadata_keys": {
                            "type": "string",
                            "description": "Comma-separated list of metadata keys to include (optional, includes all if not specified)",
                        },
                    },
                },
            ),
        ]
    )

    return tools


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls."""
    global _corpus, _text_analyzer, _csv_analyzer, _ml_analyzer

    try:
        # Corpus Management Tools
        if name == "load_corpus":
            inp = arguments.get("inp")
            source = arguments.get("source")
            text_columns = arguments.get("text_columns", "")
            ignore_words = arguments.get("ignore_words", "")

            if _init_corpus(inp, source, text_columns, ignore_words):
                doc_count = len(_corpus.documents) if _corpus else 0
                return [
                    TextContent(
                        type="text",
                        text=f"Corpus loaded successfully with {doc_count} document(s)",
                    )
                ]
            else:
                return [TextContent(type="text", text="Failed to load corpus")]

        elif name == "save_corpus":
            if not _corpus:
                return [TextContent(type="text", text="No corpus loaded")]

            out = arguments["out"]
            read_data = ReadData(corpus=_corpus)
            read_data.write_corpus_to_json(out, corpus=_corpus)
            return [TextContent(type="text", text=f"Corpus saved to {out}")]

        elif name == "add_document":
            if not _corpus:
                return [TextContent(type="text", text="No corpus loaded")]

            from ..model.document import Document

            doc = Document(
                id=arguments["doc_id"],
                text=arguments["text"],
                name=arguments.get("name"),
                description=None,
                score=0.0,
                metadata={},
            )
            _corpus.add_document(doc)
            return [
                TextContent(type="text", text=f"Document {arguments['doc_id']} added")
            ]

        elif name == "remove_document":
            if not _corpus:
                return [TextContent(type="text", text="No corpus loaded")]

            _corpus.remove_document_by_id(arguments["doc_id"])
            return [
                TextContent(type="text", text=f"Document {arguments['doc_id']} removed")
            ]

        elif name == "get_document":
            if not _corpus:
                return [TextContent(type="text", text="No corpus loaded")]

            doc = _corpus.get_document_by_id(arguments["doc_id"])
            if doc:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(doc.model_dump(), indent=2, default=str),
                    )
                ]
            return [TextContent(type="text", text="Document not found")]

        elif name == "list_documents":
            if not _corpus:
                return [TextContent(type="text", text="No corpus loaded")]

            doc_ids = _corpus.get_all_document_ids()
            return [TextContent(type="text", text=json.dumps(doc_ids, indent=2))]

        elif name == "add_relationship":
            if not _corpus:
                return [TextContent(type="text", text="No corpus loaded")]

            _corpus.add_relationship(
                arguments["first"], arguments["second"], arguments["relation"]
            )
            return [TextContent(type="text", text="Relationship added")]

        elif name == "get_relationships":
            if not _corpus:
                return [TextContent(type="text", text="No corpus loaded")]

            rels = _corpus.get_relationships()
            return [TextContent(type="text", text=json.dumps(rels, indent=2))]

        elif name == "get_relationships_for_keyword":
            if not _corpus:
                return [TextContent(type="text", text="No corpus loaded")]

            rels = _corpus.get_all_relationships_for_keyword(arguments["keyword"])
            return [TextContent(type="text", text=json.dumps(rels, indent=2))]

        # NLP/Text Analysis Tools
        elif name == "generate_coding_dictionary":
            if not _text_analyzer:
                return [TextContent(type="text", text="No corpus loaded")]

            _text_analyzer.make_spacy_doc()
            result = _text_analyzer.print_coding_dictionary(
                num=arguments.get("num", 3), top_n=arguments.get("top_n", 3)
            )
            return [
                TextContent(type="text", text=json.dumps(result, indent=2, default=str))
            ]

        elif name == "topic_modeling":
            if not _corpus:
                return [TextContent(type="text", text="No corpus loaded")]

            cluster = Cluster(corpus=_corpus)
            cluster.build_lda_model(topics=arguments.get("num_topics", 3))
            result = cluster.print_topics(num_words=arguments.get("num_words", 5))
            return [
                TextContent(type="text", text=json.dumps(result, indent=2, default=str))
            ]

        elif name == "assign_topics":
            if not _corpus:
                return [TextContent(type="text", text="No corpus loaded")]

            cluster = Cluster(corpus=_corpus)
            cluster.build_lda_model(topics=arguments.get("num_topics", 3))
            result = cluster.format_topics_sentences(visualize=False)
            return [
                TextContent(type="text", text=json.dumps(result, indent=2, default=str))
            ]

        elif name == "extract_categories":
            if not _text_analyzer:
                return [TextContent(type="text", text="No corpus loaded")]

            _text_analyzer.make_spacy_doc()
            result = _text_analyzer.print_categories(num=arguments.get("num", 10))
            return [
                TextContent(type="text", text=json.dumps(result, indent=2, default=str))
            ]

        elif name == "generate_summary":
            if not _text_analyzer:
                return [TextContent(type="text", text="No corpus loaded")]

            _text_analyzer.make_spacy_doc()
            result = _text_analyzer.generate_summary(weight=arguments.get("weight", 10))
            return [TextContent(type="text", text=str(result))]

        elif name == "sentiment_analysis":
            if not _corpus:
                return [TextContent(type="text", text="No corpus loaded")]

            sentiment = Sentiment(corpus=_corpus)
            result = sentiment.get_sentiment(
                documents=arguments.get("documents", False),
                verbose=arguments.get("verbose", True),
            )
            return [TextContent(type="text", text=str(result))]

        # Text/Corpus filtering tools
        elif name == "filter_documents":
            if not _text_analyzer:
                return [TextContent(type="text", text="No corpus loaded")]

            metadata_key = arguments.get("metadata_key", "keywords")
            metadata_value = arguments.get("metadata_value")
            if not metadata_value:
                return [TextContent(type="text", text="metadata_value is required")]

            msg = _text_analyzer.filter_documents(
                metadata_key=metadata_key, metadata_value=metadata_value, mcp=True
            )
            return [TextContent(type="text", text=str(msg))]

        elif name == "document_count":
            if not _text_analyzer:
                return [TextContent(type="text", text="No corpus loaded")]

            try:
                count = _text_analyzer.document_count()
            except Exception as e:
                return [TextContent(type="text", text=f"Error: {e}")]
            return [TextContent(type="text", text=f"Document count: {count}")]

        # DataFrame/CSV Tools
        elif name == "get_df_columns":
            if not _corpus:
                return [TextContent(type="text", text="No corpus loaded")]

            cols = _corpus.get_all_df_column_names()
            return [TextContent(type="text", text=json.dumps(cols, indent=2))]

        elif name == "get_df_row_count":
            if not _corpus:
                return [TextContent(type="text", text="No corpus loaded")]

            count = _corpus.get_row_count()
            return [TextContent(type="text", text=f"Row count: {count}")]

        elif name == "get_df_row":
            if not _corpus:
                return [TextContent(type="text", text="No corpus loaded")]

            row = _corpus.get_row_by_index(arguments["index"])
            if row is not None:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(row.to_dict(), indent=2, default=str),
                    )
                ]
            return [TextContent(type="text", text="Row not found")]

        # CSV Column/DataFrame operations
        elif name == "bin_a_column":
            if not _csv_analyzer:
                return [TextContent(type="text", text="No CSV data available")]

            msg = _csv_analyzer.bin_a_column(
                column_name=arguments["column_name"], bins=arguments.get("bins", 2)
            )
            return [TextContent(type="text", text=str(msg))]

        elif name == "one_hot_encode_column":
            if not _csv_analyzer:
                return [TextContent(type="text", text="No CSV data available")]

            msg = _csv_analyzer.one_hot_encode_column(
                column_name=arguments["column_name"]
            )
            return [TextContent(type="text", text=str(msg))]

        elif name == "filter_rows_by_column_value":
            if not _csv_analyzer:
                return [TextContent(type="text", text="No CSV data available")]

            msg = _csv_analyzer.filter_rows_by_column_value(
                column_name=arguments["column_name"], value=arguments["value"], mcp=True
            )
            return [TextContent(type="text", text=str(msg))]

        elif name == "oversample":
            if not _csv_analyzer:
                return [TextContent(type="text", text="No CSV data available")]

            result = _csv_analyzer.oversample(mcp=True)
            return [TextContent(type="text", text=str(result))]

        elif name == "restore_oversample":
            if not _csv_analyzer:
                return [TextContent(type="text", text="No CSV data available")]

            result = _csv_analyzer.restore_oversample(mcp=True)
            return [TextContent(type="text", text=str(result))]

        elif name == "get_column_types":
            if not _csv_analyzer:
                return [TextContent(type="text", text="No CSV data available")]

            types = _csv_analyzer.get_column_types()
            return [
                TextContent(type="text", text=json.dumps(types, indent=2, default=str))
            ]

        elif name == "get_column_values":
            if not _csv_analyzer:
                return [TextContent(type="text", text="No CSV data available")]

            values = _csv_analyzer.get_column_values(arguments["column_name"])
            return [
                TextContent(type="text", text=json.dumps(values, indent=2, default=str))
            ]

        elif name == "retain_numeric_columns_only":
            if not _csv_analyzer:
                return [TextContent(type="text", text="No CSV data available")]

            _csv_analyzer.retain_numeric_columns_only()
            return [TextContent(type="text", text="Retained numeric columns only.")]

        # ML Tools
        elif name == "kmeans_clustering":
            if not _csv_analyzer:
                return [TextContent(type="text", text="No CSV data available")]
            else:
                if "include" in arguments:
                    _csv_analyzer.comma_separated_include_columns(
                        arguments.get("include") + "," + arguments.get("outcome", "")
                    )

            if not ML_AVAILABLE:
                return [TextContent(type="text", text="ML dependencies not available")]

            _csv_analyzer.retain_numeric_columns_only()

            _csv_analyzer.drop_na()
            ml = ML(csv=_csv_analyzer)
            result = ml.get_kmeans(
                number_of_clusters=arguments.get("num_clusters", 3),
                verbose=False,
                mcp=True,
            )
            return [TextContent(type="text", text=str(result))]

        elif name == "decision_tree_classification":
            if not _csv_analyzer:
                return [TextContent(type="text", text="No CSV data available")]
            else:
                if "include" in arguments:
                    _csv_analyzer.comma_separated_include_columns(
                        arguments.get("include") + "," + arguments.get("outcome", "")
                    )

            if not ML_AVAILABLE:
                return [TextContent(type="text", text="ML dependencies not available")]

            if not _ml_analyzer:
                _ml_analyzer = ML(csv=_csv_analyzer)

            result = _ml_analyzer.get_decision_tree_classes(
                y=arguments["outcome"], top_n=arguments.get("top_n", 10), mcp=True
            )
            return [TextContent(type="text", text=str(result))]

        elif name == "svm_classification":
            if not _csv_analyzer:
                return [TextContent(type="text", text="No CSV data available")]
            else:
                if "include" in arguments:
                    _csv_analyzer.comma_separated_include_columns(
                        arguments.get("include") + "," + arguments.get("outcome", "")
                    )

            if not ML_AVAILABLE:
                return [TextContent(type="text", text="ML dependencies not available")]

            if not _ml_analyzer:
                _ml_analyzer = ML(csv=_csv_analyzer)

            result = _ml_analyzer.svm_confusion_matrix(
                y=arguments["outcome"], test_size=0.25, mcp=True
            )
            return [TextContent(type="text", text=str(result))]

        elif name == "neural_network_classification":
            if not _csv_analyzer:
                return [TextContent(type="text", text="No CSV data available")]
            else:
                if "include" in arguments:
                    _csv_analyzer.comma_separated_include_columns(
                        arguments.get("include") + "," + arguments.get("outcome", "")
                    )

            if not ML_AVAILABLE:
                return [TextContent(type="text", text="ML dependencies not available")]

            if not _ml_analyzer:
                _ml_analyzer = ML(csv=_csv_analyzer)

            result = _ml_analyzer.get_nnet_predictions(y=arguments["outcome"], mcp=True)
            return [TextContent(type="text", text=str(result))]

        elif name == "regression_analysis":
            if not _csv_analyzer:
                return [TextContent(type="text", text="No CSV data available")]
            else:
                if "include" in arguments:
                    _csv_analyzer.comma_separated_include_columns(
                        arguments.get("include") + "," + arguments.get("outcome", "")
                    )

            if not ML_AVAILABLE:
                return [TextContent(type="text", text="ML dependencies not available")]

            if not _ml_analyzer:
                _ml_analyzer = ML(csv=_csv_analyzer)

            result = _ml_analyzer.get_regression(y=arguments["outcome"], mcp=True)
            return [TextContent(type="text", text=str(result))]

        elif name == "pca_analysis":
            if not _csv_analyzer:
                return [TextContent(type="text", text="No CSV data available")]
            else:
                if "include" in arguments:
                    _csv_analyzer.comma_separated_include_columns(
                        arguments.get("include") + "," + arguments.get("outcome", "")
                    )

            if not ML_AVAILABLE:
                return [TextContent(type="text", text="ML dependencies not available")]

            if not _ml_analyzer:
                _ml_analyzer = ML(csv=_csv_analyzer)

            result = _ml_analyzer.get_pca(
                y=arguments["outcome"], n=arguments.get("n_components", 3), mcp=True
            )
            return [TextContent(type="text", text=str(result))]

        elif name == "association_rules":
            if not _csv_analyzer:
                return [TextContent(type="text", text="No CSV data available")]
            else:
                if "include" in arguments:
                    _csv_analyzer.comma_separated_include_columns(
                        arguments.get("include") + "," + arguments.get("outcome", "")
                    )

            if not ML_AVAILABLE:
                return [TextContent(type="text", text="ML dependencies not available")]

            if not _ml_analyzer:
                _ml_analyzer = ML(csv=_csv_analyzer)

            min_support = arguments.get("min_support", 50) / 100
            min_threshold = arguments.get("min_threshold", 50) / 100

            result = _ml_analyzer.get_apriori(
                y=arguments["outcome"],
                min_support=min_support,
                min_threshold=min_threshold,
                mcp=True,
            )
            return [TextContent(type="text", text=str(result))]

        elif name == "knn_search":
            if not _csv_analyzer:
                return [TextContent(type="text", text="No CSV data available")]
            else:
                if "include" in arguments:
                    _csv_analyzer.comma_separated_include_columns(
                        arguments.get("include") + "," + arguments.get("outcome", "")
                    )

            if not ML_AVAILABLE:
                return [TextContent(type="text", text="ML dependencies not available")]

            if not _ml_analyzer:
                _ml_analyzer = ML(csv=_csv_analyzer)

            result = _ml_analyzer.knn_search(
                y=arguments["outcome"],
                n=arguments.get("n", 3),
                r=arguments.get("record", 1),
                mcp=True,
            )
            return [TextContent(type="text", text=str(result))]

        elif name == "lstm_text_classification":
            if not _csv_analyzer:
                return [TextContent(type="text", text="No CSV data available")]

            if not ML_AVAILABLE:
                return [TextContent(type="text", text="ML dependencies not available")]

            if not _ml_analyzer:
                _ml_analyzer = ML(csv=_csv_analyzer)

            result = _ml_analyzer.get_lstm_predictions(y=arguments["outcome"], mcp=True)
            return [TextContent(type="text", text=str(result))]

        elif name == "reset_corpus_state":
            _corpus = None
            _text_analyzer = None
            _csv_analyzer = None
            _ml_analyzer = None
            return [
                TextContent(type="text", text="Global corpus state has been reset.")
            ]

        elif name == "semantic_search":
            if not _corpus:
                return [
                    TextContent(
                        type="text", text="No corpus loaded. Use load_corpus first."
                    )
                ]

            try:
                from ..semantic import Semantic

                query = arguments.get("query")
                if not query:
                    return [TextContent(type="text", text="query is required")]

                n_results = arguments.get("n_results", 5)

                semantic_analyzer = Semantic(_corpus)
                result_corpus = semantic_analyzer.get_similar(
                    query, n_results=n_results
                )

                # Update global corpus
                # global _corpus
                _corpus = result_corpus

                # Prepare response
                response_text = f"Semantic search completed for query: '{query}'\n"
                response_text += (
                    f"Found {len(result_corpus.documents)} similar documents\n\n"
                )
                response_text += "Document IDs:\n"
                for doc in result_corpus.documents[:10]:  # Show first 10
                    response_text += f"- {doc.id}: {doc.name or 'No name'}\n"
                if len(result_corpus.documents) > 10:
                    response_text += (
                        f"... and {len(result_corpus.documents) - 10} more\n"
                    )

                return [TextContent(type="text", text=response_text)]

            except ImportError:
                return [
                    TextContent(
                        type="text",
                        text="chromadb is not installed. Install with: pip install chromadb",
                    )
                ]
            except Exception as e:
                return [
                    TextContent(type="text", text=f"Error during semantic search: {e}")
                ]

        elif name == "find_similar_documents":
            if not _corpus:
                return [
                    TextContent(
                        type="text", text="No corpus loaded. Use load_corpus first."
                    )
                ]

            try:
                from ..semantic import Semantic

                document_ids = arguments.get("document_ids")
                if not document_ids:
                    return [TextContent(type="text", text="document_ids is required")]

                n_results = arguments.get("n_results", 5)
                threshold = arguments.get("threshold", 0.7)

                semantic_analyzer = Semantic(_corpus)
                similar_doc_ids = semantic_analyzer.get_similar_documents(
                    document_ids=document_ids,
                    n_results=n_results,
                    threshold=threshold
                )

                # Prepare response
                response_text = f"Finding documents similar to: '{document_ids}'\n"
                response_text += f"Number of results requested: {n_results}\n"
                response_text += f"Similarity threshold: {threshold}\n"
                response_text += f"Found {len(similar_doc_ids)} similar documents\n\n"

                if similar_doc_ids:
                    response_text += "Similar Document IDs:\n"
                    for doc_id in similar_doc_ids:
                        doc = _corpus.get_document_by_id(doc_id)
                        doc_name = f" - {doc.name}" if doc and doc.name else ""
                        response_text += f"  • {doc_id}{doc_name}\n"

                    response_text += "\nThis feature is useful for:\n"
                    response_text += "- Literature reviews: Find additional relevant papers\n"
                    response_text += "- Qualitative research: Identify documents with similar themes\n"
                    response_text += "- Content grouping: Group similar documents for analysis\n"
                else:
                    response_text += "No similar documents found above the threshold.\n"
                    response_text += "Try lowering the threshold or using different reference documents.\n"

                return [TextContent(type="text", text=response_text)]

            except ImportError:
                return [
                    TextContent(
                        type="text",
                        text="chromadb is not installed. Install with: pip install chromadb",
                    )
                ]
            except Exception as e:
                return [
                    TextContent(type="text", text=f"Error finding similar documents: {e}")
                ]

        elif name == "semantic_chunk_search":
            if not _corpus:
                return [
                    TextContent(
                        type="text", text="No corpus loaded. Use load_corpus first."
                    )
                ]

            try:
                from ..semantic import Semantic

                query = arguments.get("query")
                doc_id = arguments.get("doc_id")

                if not query:
                    return [TextContent(type="text", text="query is required")]
                if not doc_id:
                    return [TextContent(type="text", text="doc_id is required")]

                threshold = arguments.get("threshold", 0.5)
                n_results = arguments.get("n_results", 10)

                semantic_analyzer = Semantic(_corpus)
                chunks = semantic_analyzer.get_similar_chunks(
                    query=query,
                    doc_id=doc_id,
                    threshold=threshold,
                    n_results=n_results
                )

                # Prepare response
                response_text = f"Semantic chunk search completed for query: '{query}'\n"
                response_text += f"Document ID: {doc_id}\n"
                response_text += f"Threshold: {threshold}\n"
                response_text += f"Found {len(chunks)} matching chunks\n\n"

                if chunks:
                    response_text += "Matching chunks:\n"
                    response_text += "=" * 60 + "\n\n"
                    for i, chunk in enumerate(chunks, 1):
                        response_text += f"Chunk {i}:\n"
                        response_text += chunk + "\n"
                        response_text += "-" * 60 + "\n\n"

                    response_text += f"\nThese {len(chunks)} chunks can be used for coding/annotating the document.\n"
                    response_text += "You can adjust the threshold to get more or fewer results.\n"
                else:
                    response_text += "No chunks matched the query above the threshold.\n"
                    response_text += "Try lowering the threshold or use a different query.\n"

                return [TextContent(type="text", text=response_text)]

            except ImportError:
                return [
                    TextContent(
                        type="text",
                        text="chromadb is not installed. Install with: pip install chromadb",
                    )
                ]
            except Exception as e:
                return [
                    TextContent(type="text", text=f"Error during semantic chunk search: {e}")
                ]

        elif name == "export_metadata_df":
            if not _corpus:
                return [
                    TextContent(
                        type="text", text="No corpus loaded. Use load_corpus first."
                    )
                ]

            try:
                from ..semantic import Semantic

                metadata_keys_str = arguments.get("metadata_keys")
                metadata_keys = None
                if metadata_keys_str:
                    metadata_keys = [k.strip() for k in metadata_keys_str.split(",")]

                semantic_analyzer = Semantic(_corpus)
                result_corpus = semantic_analyzer.get_df(metadata_keys=metadata_keys)

                # Update global corpus
                # global _corpus
                _corpus = result_corpus

                # Prepare response
                if result_corpus.df is not None:
                    response_text = "Metadata exported to DataFrame\n"
                    response_text += f"Shape: {result_corpus.df.shape}\n"
                    response_text += f"Columns: {list(result_corpus.df.columns)}\n\n"
                    response_text += "First 5 rows:\n"
                    response_text += result_corpus.df.head().to_string()
                    return [TextContent(type="text", text=response_text)]
                else:
                    return [TextContent(type="text", text="No DataFrame created")]

            except ImportError:
                return [
                    TextContent(
                        type="text",
                        text="chromadb is not installed. Install with: pip install chromadb",
                    )
                ]
            except Exception as e:
                return [TextContent(type="text", text=f"Error exporting metadata: {e}")]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]


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
                        text="""# CRISP-T Analysis Workflow

Follow these steps to conduct a comprehensive analysis:

## Phase 1: Data Preparation and Exploration

1. **Load your data**
   - Use `load_corpus` tool with either `inp` (existing corpus) or `source` (directory/URL)
   - For CSV data with text columns, specify `text_columns` parameter

2. **Inspect the data**
   - Use `list_documents` to see all documents
   - Use `get_df_columns` and `get_df_row_count` if you have numeric data
   - Use `get_document` to examine specific documents

## Phase 2: Descriptive Analysis

3. **Generate coding dictionary**
   - Use `generate_coding_dictionary` with appropriate `num` and `top_n` parameters
   - This reveals categories (verbs), properties (nouns), and dimensions (adjectives)

4. **Perform sentiment analysis**
   - Use `sentiment_analysis` to understand emotional tone
   - Set `documents=true` for document-level analysis

5. **Basic statistical exploration**
   - Use `get_df_row` to examine specific data points
   - Review column distributions

## Phase 3: Advanced Pattern Discovery

6. **Topic modeling**
   - Use `topic_modeling` to discover latent themes (set appropriate `num_topics`)
   - Use `assign_topics` to assign documents to their dominant topics
   - Topics generate keywords that can be used to categorize documents

7. **Numerical clustering** (if you have numeric data)
   - Use `kmeans_clustering` to segment your data
   - Review cluster profiles to understand groupings

8. **Association rules** (if applicable)
   - Use `extract_categories` for text-based associations
   - Use `association_rules` for numeric pattern mining

## Phase 4: Predictive Modeling (if you have an outcome variable)

9. **Classification**
   - Use `decision_tree_classification` to get feature importance rankings
   - Use `svm_classification` for robust classification
   - Use `neural_network_classification` for complex patterns

10. **Regression analysis**
    - Use `regression_analysis` to understand factor relationships
    - It auto-detects binary outcomes (logistic) vs continuous (linear)
    - Returns coefficients showing strength and direction of relationships

11. **Dimensionality reduction**
    - Use `pca_analysis` to reduce feature space

## Phase 5: Validation and Triangulation

12. **Create relationships**
    - Use `add_relationship` to link text keywords (from topics) with numeric columns
    - Example: link topic keywords to demographic or outcome variables
    - Use format like: first="text:healthcare", second="num:age_group", relation="correlates"

13. **Validate findings**
    - Compare topic assignments with numerical clusters
    - Validate sentiment patterns with outcome variables
    - Use `get_relationships_for_keyword` to explore connections

14. **Save your work**
    - Use `save_corpus` to persist all analyses and metadata
    - The corpus retains all transformations and relationships

## Tips
- Always load corpus first
- Topic modeling creates keywords useful for filtering/categorizing documents
- Decision trees and regression provide variable importance and coefficients
- Link text findings (topics) with numeric data using relationships
- Save frequently to preserve your analysis state
""",
                    ),
                )
            ],
        )

    elif name == "triangulation_guide":
        return GetPromptResult(
            description="Guide for triangulating findings",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text="""# Triangulation Guide for CRISP-T

## What is Triangulation?

Triangulation involves validating findings by comparing and contrasting results from different analytical methods or data sources. In CRISP-T, this means linking textual insights with numeric patterns.

## Key Strategies

### 1. Link Topic Keywords to Variables

After topic modeling:
- Topics generate keywords representing themes
- Use `add_relationship` to link keywords to relevant dataframe columns
- Example: If topic discusses "satisfaction", link to satisfaction score column

### 2. Compare Patterns

- Cross-reference sentiment with numeric outcomes
- Compare topic distributions across demographic groups
- Validate clustering results using both text and numbers

### 3. Use Relationships

- `add_relationship("text:keyword", "num:column", "correlates")`
- `get_relationships_for_keyword` to explore connections
- Document theoretical justifications for relationships

### 4. Validate Findings

- Check if text-based themes align with numeric clusters
- Test if sentiment patterns predict outcomes
- Use regression to quantify relationships
- Decision trees reveal which factors matter most

## Example Workflow

1. Topic model reveals "healthcare access" theme
2. Assign documents to topics (creates keyword labels)
3. Link "healthcare access" keyword to "insurance_status" column
4. Run regression with insurance_status as outcome
5. Compare topic prevalence across insurance groups
6. Add relationships to document connections
7. Validate using classification models

## Best Practices

- Document all relationships you create
- Test relationships statistically
- Use multiple analytical approaches
- Save corpus frequently to preserve metadata
- Revisit and refine relationships as analysis progresses
""",
                    ),
                )
            ],
        )

    raise ValueError(f"Unknown prompt: {name}")


async def main():
    """Main entry point for the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())
