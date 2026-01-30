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

from ..cluster import Cluster
from ..helpers.analyzer import get_csv_analyzer, get_text_analyzer
from ..helpers.clib import clear_cache
from ..helpers.initializer import initialize_corpus
from ..read_data import ReadData
from ..sentiment import Sentiment
from .utils.responses import (
    error_response,
    no_corpus_response,
    no_csv_analyzer_response,
    success_response,
)

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
    """List available tools."""
    tools = [
        # Corpus management tools
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
        # ! NLP/Text Analysis Tools
        Tool(
            name="generate_coding_dictionary",
            description="""
            Generate a qualitative coding dictionary with categories organized by grammatical function:
            - Verbs (actions/processes)
            - Nouns (properties/concepts)
            - Adjectives/Adverbs (dimensions/qualities)

            Reveals main themes and constructs in corpus. Excellent for grounded theory and thematic analysis.
            IMPORTANT: This analyzes CORPUS-LEVEL patterns. Use assign_topics for document-level coding.

            Configuration tips:
            - num: Increase (5-10) for exploratory analysis, decrease (3) for focused analysis
            - top_n: Show 3-5 items per category for balanced view
            - ignore: Exclude stop words and domain-specific noise words
            - filters: Use key=value to analyze subsets (e.g., sentiment=positive)

            Workflow: Usually second step after load_corpus for understanding corpus structure.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "num": {
                        "type": "integer",
                        "description": "Number of categories to extract",
                        "default": 3,
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Top N items per category",
                        "default": 3,
                    },
                    "ignore": {
                        "type": "array",
                        "description": "List of words to ignore",
                        "items": {"type": "string"},
                    },
                    "filters": {
                        "type": "array",
                        "description": "Filters to apply on documents (key=value or key:value)",
                        "items": {"type": "string"},
                    },
                },
            },
        ),
        Tool(
            name="topic_modeling",
            description="""
            Discover latent topics in corpus using Latent Dirichlet Allocation (LDA). Returns probabilistic topic distributions.
            Each topic is represented as a weighted set of keywords, useful for understanding corpus themes and patterns.

            IMPORTANT: This discovers corpus-level topics. Follow with assign_topics to label documents with their dominant topics.

            Parameter guidance:
            - num_topics: Start with 3-5 for initial exploration. Increase for large/diverse corpora.
              - Low (2-3): High-level themes
              - Medium (5-10): Detailed topic breakdown
              - High (15+): Fine-grained distinction (needs large corpus)
            - num_words: 5-10 recommended for interpretability

            Workflow:
            1. Run topic_modeling to discover topics
            2. Review topic keywords to validate they're meaningful
            3. Use assign_topics to assign documents to topics
            4. Add relationships linking topics to numeric outcomes
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "num_topics": {
                        "type": "integer",
                        "description": "Number of topics to generate",
                        "default": 3,
                    },
                    "num_words": {
                        "type": "integer",
                        "description": "Number of words per topic",
                        "default": 5,
                    },
                },
            },
        ),
        Tool(
            name="assign_topics",
            description="""
            Assign each document to its dominant topics with contribution percentages. Provides document-level topic labels.
            Results can be used for filtering, categorization, and adding relationships to numeric outcomes.

            IMPORTANT CACHE BEHAVIOR:
            - First run creates cache (may take time for large corpora)
            - Subsequent runs use cache (fast)
            - When changing filters: MUST call clear_cache first, then rerun
            - If you change num_topics after initial analysis: clear_cache first

            Workflow:
            1. Run topic_modeling first (discovers corpus topics)
            2. Call assign_topics to label documents
            3. Use results in filter_documents or add_relationship

            Tip: Check topic keywords from topic_modeling before assigning to ensure they're meaningful.
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
            Extract distinct categories/concepts from corpus as weighted bag-of-terms. Similar to topic modeling but
            provides category-level (rather than document-level) analysis.

            Use for: Quick overview of major concepts, validation of topic modeling results, creating concept hierarchies, understanding corpus vocabulary.

            Configuration:
            - num: 5-15 recommended. Higher values reveal more fine-grained distinctions.

            Comparison to topic_modeling:
            - extract_categories: Faster, corpus-level only, simpler interpretation
            - topic_modeling: Probabilistic, document-level mapping possible
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
            Generate extractive summary (key sentences from original documents) representing entire corpus.
            Useful for quick overviews, stakeholder reports, and understanding dominant themes.

            Use for: Executive summaries of corpus content, understanding key passages, report generation, identifying representative quotes.

            Configuration:
            - weight: 5-15 for most corpora
              - Low (5): Concise 1-2 sentence summary
              - Medium (10): Balanced overview
              - High (20+): Comprehensive summary with many key points

            Note: This is extractive (using original sentences), not generative (creating new text).
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
            Analyze emotional tone in corpus using VADER (Valence Aware Dictionary and sEntiment Reasoner).
            Returns: positive, negative, neutral proportions + compound sentiment score (-1 to +1).

            VADER is optimized for:
            - Social media text and informal language
            - Texts with emojis, contractions, slang
            - Mixed sentiment (not strictly positive/negative)

            Output options:
            - documents=false (default): Corpus-level sentiment (useful for: Overall tone, trend analysis, outcome prediction)
            - documents=true: Document-level sentiment (useful for: Tracking individual perspectives, document categorization)

            Workflow:
            1. Run sentiment_analysis(documents=false) for corpus overview
            2. If interesting pattern found, run with documents=true to drill down
            3. Use results to add relationships: text:sentiment_category|num:outcome_metric|correlates

            Tip: For more rigorous NLP, combine with topic_modeling for aspect-based sentiment.
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
            description=(
                "Filter corpus documents based on coding links and/or metadata filters. "
                "Supports metadata filters (key:value or key=value) and link filters for embedding and temporal relationships. "
                "Link format: 'embedding:text' (documents with embedding links), 'embedding:df' (matches temporal links), etc. "
                "Apply AND logic when combining filters. Returns filtered corpus and document count. "
                "Use to subset corpus for sub-analysis, identify documents with specific relationships, or validate link creation. "
                "Tip: Use list_documents first to understand existing links before filtering. "
                "Workflow: Filter to documents with specific coded categories → analyze subset → save as branch for comparison. "
                "Updates the active corpus."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "string",
                        "description": (
                            "Filter string. Examples: 'keywords=health', 'embedding:text', 'embedding:df', 'temporal:text', 'temporal:df'. "
                            "Multiple filters can be applied sequentially."
                        ),
                    },
                },
                "required": ["filter"],
            },
        ),
        Tool(
            name="document_count",
            description="Return number of documents currently in active corpus (accounting for any active filters). Use to validate filtering operations or understand corpus size after subsetting. Useful in workflows: load corpus → filter by criteria → check count to verify expected subset size.",
            inputSchema={"type": "object", "properties": {}},
        ),
        # DataFrame/CSV Tools
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
        # CSV Column/DataFrame operations
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
        Tool(
            name="reset_corpus_state",
            description="Reset the global corpus, text analyzer, and CSV analyzer state. Clear all loaded data and start fresh.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="clear_cache",
            description="Delete the cache folder if it exists. Use this to clear cached analysis results and free up disk space.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]

    # Add ML tools if available
    if ML_AVAILABLE:
        tools.extend(
            [
                Tool(
                    name="kmeans_clustering",
                    description="""
                Perform K-Means clustering on numeric features to segment data into groups. Essential for: Unsupervised exploratory analysis, Finding natural groupings in data, Creating clusters for mixed-methods linking to text themes.

                Workflow: get_df_columns → retain_numeric_columns_only (if needed) → kmeans_clustering (start with num_clusters=3) → use cluster assignments to add_relationship linking clusters to text topics or coded categories.

                Parameters:
                - num_clusters: Start with 3-5 for exploratory analysis; use elbow method (try 2-10) to find optimal k
                - include: Specify numeric columns for clustering (e.g., "age,income,satisfaction")
                - outcome: Optional column to exclude from clustering features

                Tip: Normalize/scale columns first for best results; clustering is sensitive to feature magnitude.
                """,
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "num_clusters": {
                                "type": "integer",
                                "description": "Number of clusters (default: 3, typical range: 2-10)",
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
                Train decision tree classifier to identify predictive features. Returns variable importance rankings. Essential for: Understanding feature importance (what predicts outcome?), Creating interpretable ML models, Validating qualitative coding against numeric outcomes.

                Workflow: filter_documents + filter_rows_by_column_value (subset to key groups) → decision_tree_classification (outcome=target_column) → Examine top features → Add relationships linking top predictors to text themes.

                Parameters:
                - outcome: Target variable (DataFrame column or text metadata field if linkage_method specified)
                - include: Feature columns for model (comma-separated)
                - top_n: Number of important features to return (default: 10, typical: 5-20)
                - linkage_method: For text metadata outcomes: "id" (document level), "embedding" (semantic), "temporal" (time-based), "keyword" (link-based)
                - aggregation: When multiple documents per outcome: "majority" (most common), "mean" (average), "mode", "first"

                Tip: Binary classification (2 classes) more reliable than multi-class; start simple.
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
                                "description": "Top N important features to return (default: 10, typical: 5-20)",
                                "default": 10,
                            },
                            "include": {
                                "type": "string",
                                "description": "Comma-separated columns to include",
                            },
                            "linkage_method": {
                                "type": "string",
                                "description": "Linkage method for text metadata outcomes: id, embedding, temporal, keyword",
                                "enum": ["id", "embedding", "temporal", "keyword"],
                            },
                            "aggregation": {
                                "type": "string",
                                "description": "Aggregation strategy for multiple documents: majority, mean, first, mode",
                                "enum": ["majority", "mean", "first", "mode"],
                                "default": "majority",
                            },
                        },
                        "required": ["outcome", "include"],
                    },
                ),
                Tool(
                    name="svm_classification",
                    description="""
                Perform SVM (Support Vector Machine) classification. Returns confusion matrix and accuracy. Essential for: Binary/multiclass classification problems, Finding decision boundaries in high-dimensional data, Validating text coding against numeric outcomes.

                Workflow: prepare numeric features → bin outcome if needed (e.g., satisfaction_score → high/low) → svm_classification → validate results → create relationships linking predictions to text themes.

                Parameters:
                - outcome: Target variable (DataFrame column or text metadata)
                - include: Feature columns (comma-separated)
                - linkage_method: For text outcomes: "id", "embedding", "temporal", "keyword"
                - aggregation: Strategy for multiple documents

                Use cases:
                - Binary (2 classes): Most reliable, typical use case
                - Multiclass (3+ classes): Possible but more challenging

                Compare to: decision_tree_classification (more interpretable) vs svm_classification (better for complex boundaries)
                Tip: Normalize/scale features for better SVM performance.
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
                            "linkage_method": {
                                "type": "string",
                                "description": "Linkage method for text metadata outcomes: id, embedding, temporal, keyword",
                                "enum": ["id", "embedding", "temporal", "keyword"],
                            },
                            "aggregation": {
                                "type": "string",
                                "description": "Aggregation strategy for multiple documents",
                                "enum": ["majority", "mean", "first", "mode"],
                                "default": "majority",
                            },
                        },
                        "required": ["outcome", "include"],
                    },
                ),
                Tool(
                    name="neural_network_classification",
                    description="""
                Train neural network (deep learning) classifier for complex pattern detection. Returns predictions and accuracy. Best for: Large datasets (1000+ rows), Complex non-linear relationships, Multiclass problems (3+ outcomes).

                Workflow: prepare data with bin_a_column (categorize outcome) → one_hot_encode_column (for features) → neural_network_classification → evaluate results.

                Parameters:
                - outcome: Target variable (binary or multiclass)
                - include: Feature columns (comma-separated)
                - linkage_method/aggregation: Same as SVM

                Warning: Requires more data than decision_tree or SVM. Small datasets (<100 rows) may overfit.

                Compare to: decision_tree (interpretable), svm (good baseline), neural_network (handles complex patterns).
                Tip: Start with simpler models (decision_tree) first; use neural networks when simpler models underperform.
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
                            "linkage_method": {
                                "type": "string",
                                "description": "Linkage method for text metadata outcomes",
                                "enum": ["id", "embedding", "temporal", "keyword"],
                            },
                            "aggregation": {
                                "type": "string",
                                "description": "Aggregation strategy",
                                "enum": ["majority", "mean", "first", "mode"],
                                "default": "majority",
                            },
                        },
                        "required": ["outcome", "include"],
                    },
                ),
                Tool(
                    name="regression_analysis",
                    description="""
                Perform linear (numeric outcome) or logistic (binary outcome) regression. Returns coefficients showing relationship strength/direction for each predictor. Essential for: Testing hypotheses about what predicts outcome, Quantifying predictor effects (which factors matter most?), Validating relationships found in text analysis.

                Workflow: filter by groups → regression_analysis(outcome=target, include="factor1,factor2") → Extract coefficients → Add relationships linking significant factors to text themes.

                Auto-detects regression type:
                - Numeric outcome: Linear regression (continuous prediction)
                - Binary/categorical: Logistic regression (probability prediction)

                Parameters:
                - outcome: Target variable (numeric or binary categorical)
                - include: Predictor columns (comma-separated)
                - linkage_method: For text outcomes
                - aggregation: Default="mean" for regression (numeric aggregation)

                Interpretation: Larger coefficient = stronger effect on outcome (positive/negative direction).
                Tip: Start with top factors from decision_tree_classification for focused regression.
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
                            "linkage_method": {
                                "type": "string",
                                "description": "Linkage method for text metadata outcomes",
                                "enum": ["id", "embedding", "temporal", "keyword"],
                            },
                            "aggregation": {
                                "type": "string",
                                "description": "Aggregation strategy (default: mean for regression)",
                                "enum": ["majority", "mean", "first", "mode"],
                                "default": "mean",
                            },
                        },
                        "required": ["outcome", "include"],
                    },
                ),
                Tool(
                    name="pca_analysis",
                    description="""
                Perform Principal Component Analysis for dimensionality reduction and visualization. Combines correlated features into uncorrelated principal components. Essential for: Visualizing high-dimensional data, Reducing feature count before ML (noise reduction), Exploratory analysis (which feature groups cluster together?).

                Workflow: retain_numeric_columns_only → pca_analysis(n_components=2 or 3) → visualize/create relationships linking principal components to text themes/clusters.

                Parameters:
                - n_components: Number of dimensions to keep (default: 3, typical: 2-5 for visualization, 50%+ of original features for data reduction)
                - outcome: Variable to exclude from analysis
                - include: Features for PCA (comma-separated, typically all numeric columns)

                Interpretation: Each principal component is weighted combination of original features. Explained variance % shows how much information each component captures.

                Workflow example: Do documents with topic X differ on measured variables Y,Z? PCA(n_components=2) on Y,Z → check if topic X documents separate in PCA space.
                Tip: Normalize/scale features first; PCA sensitive to feature magnitude.
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
                                "description": "Number of components (default: 3, typical: 2-5 for visualization)",
                                "default": 3,
                            },
                            "include": {
                                "type": "string",
                                "description": "Comma-separated columns to include",
                            },
                            "linkage_method": {
                                "type": "string",
                                "description": "Linkage method for text metadata outcomes",
                                "enum": ["id", "embedding", "temporal", "keyword"],
                            },
                            "aggregation": {
                                "type": "string",
                                "description": "Aggregation strategy",
                                "enum": ["majority", "mean", "first", "mode"],
                                "default": "majority",
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
                    outcome (str): The target variable (excluded from features). Can be a DataFrame column OR text metadata field (when linkage_method is specified).
                    n (int): The number of neighbors to find.
                    record (int): The record index (1-based) to find neighbors for.
                    include (str): Comma-separated columns to include.
                    linkage_method (str, optional): Linkage method when outcome is a text metadata field. Options: id, embedding, temporal, keyword.
                    aggregation (str, optional): Aggregation strategy for multiple documents per row. Options: majority, mean, first, mode.
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
                            "linkage_method": {
                                "type": "string",
                                "description": "Linkage method for text metadata outcomes",
                                "enum": ["id", "embedding", "temporal", "keyword"],
                            },
                            "aggregation": {
                                "type": "string",
                                "description": "Aggregation strategy",
                                "enum": ["majority", "mean", "first", "mode"],
                                "default": "majority",
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
                description="Perform semantic search to find documents similar to query using ChromaDB embeddings (not keyword search). Essential for: Literature reviews (find relevant papers), Qualitative coding (find documents matching themes), Exploring corpus conceptually. Returns documents ordered by semantic similarity (highest first). Query examples: 'healthcare barriers' (finds relevant passages regardless of exact wording). Compare to: filter_documents (exact metadata matching), semantic_chunk_search (searches within one document). Workflow: semantic_search(query='theme') → examine results → use top doc IDs for find_similar_documents. Tip: Refine query if results miss relevant documents.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query text (conceptual search, not keyword)",
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Number of similar documents to return (default: 5, typical: 3-20)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="find_similar_documents",
                description="Find documents semantically similar to reference documents (seed-based search). Essential for: Literature review snowballing (start with known relevant papers → find more similar ones), Validation (are these documents similar to my key examples?), Grouping (find all docs similar to this cluster). Parameters: document_ids (one or multiple), n_results (documents to return), threshold (minimum similarity 0-1, default 0.7 = high similarity). Workflow: Pick 1-3 exemplary documents → find_similar_documents → expand literature set → validate quality. Typical threshold: 0.5 (loose matching) to 0.9 (strict matching). Compare to: semantic_search (query-based), semantic_chunk_search (within-document search). Tip: Start with threshold=0.5 if getting too few results.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "document_ids": {
                            "type": "string",
                            "description": "Single document ID or comma-separated list of IDs to use as reference",
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Number of similar documents to return (default: 5, typical: 3-20)",
                            "default": 5,
                        },
                        "threshold": {
                            "type": "number",
                            "description": "Minimum similarity threshold 0-1 (default: 0.7=high, try 0.5 for loose matching, 0.9 for strict)",
                            "default": 0.7,
                        },
                    },
                    "required": ["document_ids"],
                },
            ),
            Tool(
                name="semantic_chunk_search",
                description="Perform semantic search within a single document to find relevant passages/chunks matching a concept or theme. Essential for: Qualitative coding (find mentions of concept X in document Y), Annotation (mark relevant sections for later analysis), Quote extraction (identify key passages supporting theme). Parameters: query (concept/theme), doc_id (document to search), threshold (minimum similarity 0-1, default 0.5), n_results (max chunks to retrieve). Workflow: Get doc_id with semantic_search → semantic_chunk_search(query='theme', doc_id=X) → extract matching chunks for coding/quotes. Threshold guidance: 0.3-0.5 (loose, catch all mentions), 0.7+ (strict, only high matches). Compare to: semantic_search (whole corpus), filter_documents (metadata-based). Tip: Use for manual annotation/coding validation.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (concept/theme to find within document)",
                        },
                        "doc_id": {
                            "type": "string",
                            "description": "Document ID to search within",
                        },
                        "threshold": {
                            "type": "number",
                            "description": "Minimum similarity threshold 0-1 (default: 0.5; try 0.3-0.7 range)",
                            "default": 0.5,
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Maximum chunks to retrieve before filtering (default: 10, typical: 5-20)",
                            "default": 10,
                        },
                    },
                    "required": ["query", "doc_id"],
                },
            ),
            Tool(
                name="export_metadata_df",
                description="Export ChromaDB collection metadata as DataFrame for analysis/visualization. Essential for: Analyzing document-level metadata (sources, dates, etc.), Exporting for external analysis tools, Creating reports. Parameters: metadata_keys (comma-separated keys to export, optional - all keys by default). Returns: DataFrame with document metadata. Workflow: semantic_search → export_metadata_df(metadata_keys='source,date') → analyze patterns in metadata.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "metadata_keys": {
                            "type": "string",
                            "description": "Comma-separated metadata keys to include (optional, all if not specified)",
                        },
                    },
                },
            ),
        ]
    )

    # Add TDABM tool
    tools.append(
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
    )

    # Temporal analysis tools
    tools.extend(
        [
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
    )

    # Embedding-based linking tools
    tools.extend(
        [
            Tool(
                name="embedding_link",
                description="""
            Link documents to dataframe rows using semantic embedding similarity. Essential for: Mixed-methods triangulation when explicit IDs/timestamps unavailable, Fuzzy matching based on content meaning, Validating qualitative themes against numeric patterns.

            Uses vector embeddings to create text↔numeric relationships based on semantic similarity (how conceptually similar are they?).

            Parameters:
            - similarity_metric: 'cosine' (default, most common for embeddings) or 'euclidean'
            - top_k: Number of DataFrame rows to link per document (default: 1, try 1-5)
            - threshold: Minimum similarity score 0-1 (e.g., 0.7 = high similarity; optional, filters low matches)
            - numeric_columns: Specific columns for embedding (default: all numeric)

            Workflow: Load corpus with numeric data → embedding_link(threshold=0.7, top_k=1) → get_relationships to inspect → visualize links.

            Compare to: temporal_link_by_time (uses timestamps), filter_documents (uses metadata/links).
            Tip: Start with top_k=1 and adjust based on results. Higher threshold=stricter matching.
            """,
                inputSchema={
                    "type": "object",
                    "properties": {
                        "similarity_metric": {
                            "type": "string",
                            "description": "Similarity metric: 'cosine' (default) or 'euclidean'",
                            "enum": ["cosine", "euclidean"],
                            "default": "cosine",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of top similar rows to link per document (default: 1, try 1-5)",
                            "default": 1,
                        },
                        "threshold": {
                            "type": "number",
                            "description": "Minimum similarity threshold 0-1 (e.g., 0.7 for high; optional, filters low matches)",
                        },
                        "numeric_columns": {
                            "type": "string",
                            "description": "Comma-separated numeric columns for embeddings (optional, default: all numeric)",
                        },
                    },
                },
            ),
            Tool(
                name="embedding_link_stats",
                description="""
            Get statistics about embedding-based links already created. Shows: document count linked, average similarity scores, distribution of links. Essential for: Validating embedding_link results, Understanding link quality (similarity score ranges), Reporting linking coverage.

            Returns: Number of linked documents, linking statistics (average similarity, etc.).

            Workflow: embedding_link(...) → embedding_link_stats → review statistics → if poor quality, re-run with different threshold/top_k → get_relationships for detailed links.
            Tip: Average similarity scores show link quality (0.7+ = good semantic alignment).
            """,
                inputSchema={
                    "type": "object",
                    "properties": {},
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
                return success_response(
                    f"Corpus loaded successfully with {doc_count} document(s)"
                )
            else:
                return error_response("Failed to load corpus")

        elif name == "save_corpus":
            if not _corpus:
                return no_corpus_response()

            out = arguments["out"]
            read_data = ReadData(corpus=_corpus)
            read_data.write_corpus_to_json(out, corpus=_corpus)
            return success_response(f"Corpus saved to {out}")

        elif name == "add_document":
            if not _corpus:
                return no_corpus_response()

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
            return success_response(f"Document {arguments['doc_id']} added")

        elif name == "remove_document":
            if not _corpus:
                return no_corpus_response()

            _corpus.remove_document_by_id(arguments["doc_id"])
            return success_response(f"Document {arguments['doc_id']} removed")

        elif name == "get_document":
            if not _corpus:
                return no_corpus_response()

            doc = _corpus.get_document_by_id(arguments["doc_id"])
            if doc:
                return success_response(
                    json.dumps(doc.model_dump(), indent=2, default=str)
                )
            return error_response("Document not found")

        elif name == "list_documents":
            if not _corpus:
                return no_corpus_response()

            doc_ids = _corpus.get_all_document_ids()
            return success_response(json.dumps(doc_ids, indent=2))

        elif name == "add_relationship":
            if not _corpus:
                return no_corpus_response()

            _corpus.add_relationship(
                arguments["first"], arguments["second"], arguments["relation"]
            )
            return success_response("Relationship added")

        elif name == "get_relationships":
            if not _corpus:
                return no_corpus_response()

            rels = _corpus.get_relationships()
            return success_response(json.dumps(rels, indent=2))

        elif name == "get_relationships_for_keyword":
            if not _corpus:
                return [TextContent(type="text", text="No corpus loaded")]

            rels = _corpus.get_all_relationships_for_keyword(arguments["keyword"])
            return [TextContent(type="text", text=json.dumps(rels, indent=2))]

        # NLP/Text Analysis Tools
        elif name == "generate_coding_dictionary":
            if not _text_analyzer:
                return no_corpus_response()

            _text_analyzer.make_spacy_doc()
            result = _text_analyzer.print_coding_dictionary(
                num=arguments.get("num", 3), top_n=arguments.get("top_n", 3)
            )
            return success_response(json.dumps(result, indent=2, default=str))

        elif name == "topic_modeling":
            if not _corpus:
                return no_corpus_response()

            cluster = Cluster(corpus=_corpus)
            cluster.build_lda_model(topics=arguments.get("num_topics", 3))
            result = cluster.print_topics(num_words=arguments.get("num_words", 5))
            return success_response(json.dumps(result, indent=2, default=str))

        elif name == "assign_topics":
            if not _corpus:
                return no_corpus_response()

            cluster = Cluster(corpus=_corpus)
            cluster.build_lda_model(topics=arguments.get("num_topics", 3))
            result = cluster.format_topics_sentences(visualize=False)
            return success_response(json.dumps(result, indent=2, default=str))

        elif name == "extract_categories":
            if not _text_analyzer:
                return no_corpus_response()

            _text_analyzer.make_spacy_doc()
            result = _text_analyzer.print_categories(num=arguments.get("num", 10))
            return success_response(json.dumps(result, indent=2, default=str))

        elif name == "generate_summary":
            if not _text_analyzer:
                return no_corpus_response()

            _text_analyzer.make_spacy_doc()
            result = _text_analyzer.generate_summary(weight=arguments.get("weight", 10))
            return success_response(str(result))

        elif name == "sentiment_analysis":
            if not _corpus:
                return no_corpus_response()

            sentiment = Sentiment(corpus=_corpus)
            result = sentiment.get_sentiment(
                documents=arguments.get("documents", False),
                verbose=arguments.get("verbose", True),
            )
            return success_response(str(result))

        # Text/Corpus filtering tools
        elif name == "filter_documents":
            if not _text_analyzer:
                return no_corpus_response()

            metadata_key = arguments.get("metadata_key", "keywords")
            metadata_value = arguments.get("metadata_value")
            if not metadata_value:
                return error_response("metadata_value is required")

            msg = _text_analyzer.filter_documents(
                metadata_key=metadata_key, metadata_value=metadata_value, mcp=True
            )
            return success_response(str(msg))

        elif name == "document_count":
            if not _text_analyzer:
                return no_corpus_response()

            try:
                count = _text_analyzer.document_count()
            except Exception as e:
                return error_response(str(e))
            return success_response(f"Document count: {count}")

        # DataFrame/CSV Tools
        elif name == "get_df_columns":
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

        # CSV Column/DataFrame operations
        elif name == "bin_a_column":
            if not _csv_analyzer:
                return no_csv_analyzer_response()

            msg = _csv_analyzer.bin_a_column(
                column_name=arguments["column_name"], bins=arguments.get("bins", 2)
            )
            return success_response(str(msg))

        elif name == "one_hot_encode_column":
            if not _csv_analyzer:
                return no_csv_analyzer_response()

            msg = _csv_analyzer.one_hot_encode_column(
                column_name=arguments["column_name"]
            )
            return success_response(str(msg))

        elif name == "filter_rows_by_column_value":
            if not _csv_analyzer:
                return no_csv_analyzer_response()

            msg = _csv_analyzer.filter_rows_by_column_value(
                column_name=arguments["column_name"], value=arguments["value"], mcp=True
            )
            return success_response(str(msg))

        elif name == "oversample":
            if not _csv_analyzer:
                return no_csv_analyzer_response()

            result = _csv_analyzer.oversample(mcp=True)
            return success_response(str(result))

        elif name == "restore_oversample":
            if not _csv_analyzer:
                return no_csv_analyzer_response()

            result = _csv_analyzer.restore_oversample(mcp=True)
            return success_response(str(result))

        elif name == "get_column_types":
            if not _csv_analyzer:
                return no_csv_analyzer_response()

            types = _csv_analyzer.get_column_types()
            return success_response(json.dumps(types, indent=2, default=str))

        elif name == "get_column_values":
            if not _csv_analyzer:
                return no_csv_analyzer_response()

            values = _csv_analyzer.get_column_values(arguments["column_name"])
            return success_response(json.dumps(values, indent=2, default=str))

        elif name == "retain_numeric_columns_only":
            if not _csv_analyzer:
                return no_csv_analyzer_response()

            _csv_analyzer.retain_numeric_columns_only()
            return success_response("Retained numeric columns only.")

        # ML Tools
        elif name == "kmeans_clustering":
            if not _csv_analyzer:
                return no_csv_analyzer_response()
            else:
                if "include" in arguments:
                    _csv_analyzer.comma_separated_include_columns(
                        arguments.get("include") + "," + arguments.get("outcome", "")
                    )

            if not ML_AVAILABLE:
                return error_response("ML dependencies not available")

            _csv_analyzer.retain_numeric_columns_only()

            _csv_analyzer.drop_na()
            ml = ML(csv=_csv_analyzer)
            result = ml.get_kmeans(
                number_of_clusters=arguments.get("num_clusters", 3),
                verbose=False,
                mcp=True,
            )
            return success_response(str(result))

        elif name == "decision_tree_classification":
            if not _csv_analyzer:
                return no_csv_analyzer_response()
            else:
                if "include" in arguments:
                    _csv_analyzer.comma_separated_include_columns(
                        arguments.get("include") + "," + arguments.get("outcome", "")
                    )

            if not ML_AVAILABLE:
                return error_response("ML dependencies not available")

            if not _ml_analyzer:
                _ml_analyzer = ML(csv=_csv_analyzer)

            result = _ml_analyzer.get_decision_tree_classes(
                y=arguments["outcome"],
                top_n=arguments.get("top_n", 10),
                mcp=True,
                linkage_method=arguments.get("linkage_method"),
                aggregation=arguments.get("aggregation", "majority"),
            )
            return success_response(str(result))

        elif name == "svm_classification":
            if not _csv_analyzer:
                return no_csv_analyzer_response()
            else:
                if "include" in arguments:
                    _csv_analyzer.comma_separated_include_columns(
                        arguments.get("include") + "," + arguments.get("outcome", "")
                    )

            if not ML_AVAILABLE:
                return error_response("ML dependencies not available")

            if not _ml_analyzer:
                _ml_analyzer = ML(csv=_csv_analyzer)

            linkage_method = arguments.get("linkage_method")
            aggregation = arguments.get("aggregation", "majority")

            result = _ml_analyzer.svm_confusion_matrix(
                y=arguments["outcome"],
                test_size=0.25,
                mcp=True,
                linkage_method=linkage_method,
                aggregation=aggregation,
            )
            return success_response(str(result))

        elif name == "neural_network_classification":
            if not _csv_analyzer:
                return no_csv_analyzer_response()
            else:
                if "include" in arguments:
                    _csv_analyzer.comma_separated_include_columns(
                        arguments.get("include") + "," + arguments.get("outcome", "")
                    )

            if not ML_AVAILABLE:
                return error_response("ML dependencies not available")

            if not _ml_analyzer:
                _ml_analyzer = ML(csv=_csv_analyzer)

            linkage_method = arguments.get("linkage_method")
            aggregation = arguments.get("aggregation", "majority")

            result = _ml_analyzer.get_nnet_predictions(
                y=arguments["outcome"],
                mcp=True,
                linkage_method=linkage_method,
                aggregation=aggregation,
            )
            return success_response(str(result))

        elif name == "regression_analysis":
            if not _csv_analyzer:
                return no_csv_analyzer_response()
            else:
                if "include" in arguments:
                    _csv_analyzer.comma_separated_include_columns(
                        arguments.get("include") + "," + arguments.get("outcome", "")
                    )

            if not ML_AVAILABLE:
                return error_response("ML dependencies not available")

            if not _ml_analyzer:
                _ml_analyzer = ML(csv=_csv_analyzer)

            linkage_method = arguments.get("linkage_method")
            aggregation = arguments.get(
                "aggregation", "mean"
            )  # Default to mean for regression

            result = _ml_analyzer.get_regression(
                y=arguments["outcome"],
                mcp=True,
                linkage_method=linkage_method,
                aggregation=aggregation,
            )
            return success_response(str(result))

        elif name == "pca_analysis":
            if not _csv_analyzer:
                return no_csv_analyzer_response()
            else:
                if "include" in arguments:
                    _csv_analyzer.comma_separated_include_columns(
                        arguments.get("include") + "," + arguments.get("outcome", "")
                    )

            if not ML_AVAILABLE:
                return error_response("ML dependencies not available")

            if not _ml_analyzer:
                _ml_analyzer = ML(csv=_csv_analyzer)

            result = _ml_analyzer.get_pca(
                y=arguments["outcome"],
                n=arguments.get("n_components", 3),
                mcp=True,
                linkage_method=arguments.get("linkage_method"),
                aggregation=arguments.get("aggregation", "majority"),
            )
            return success_response(str(result))

        elif name == "association_rules":
            if not _csv_analyzer:
                return no_csv_analyzer_response()
            else:
                if "include" in arguments:
                    _csv_analyzer.comma_separated_include_columns(
                        arguments.get("include") + "," + arguments.get("outcome", "")
                    )

            if not ML_AVAILABLE:
                return error_response("ML dependencies not available")

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
            return success_response(str(result))

        elif name == "knn_search":
            if not _csv_analyzer:
                return no_csv_analyzer_response()
            else:
                if "include" in arguments:
                    _csv_analyzer.comma_separated_include_columns(
                        arguments.get("include") + "," + arguments.get("outcome", "")
                    )

            if not ML_AVAILABLE:
                return error_response("ML dependencies not available")

            if not _ml_analyzer:
                _ml_analyzer = ML(csv=_csv_analyzer)

            result = _ml_analyzer.knn_search(
                y=arguments["outcome"],
                n=arguments.get("n", 3),
                r=arguments.get("record", 1),
                mcp=True,
                linkage_method=arguments.get("linkage_method"),
                aggregation=arguments.get("aggregation", "majority"),
            )
            return success_response(str(result))

        elif name == "lstm_text_classification":
            if not _csv_analyzer:
                return no_csv_analyzer_response()

            if not ML_AVAILABLE:
                return error_response("ML dependencies not available")

            if not _ml_analyzer:
                _ml_analyzer = ML(csv=_csv_analyzer)

            result = _ml_analyzer.get_lstm_predictions(y=arguments["outcome"], mcp=True)
            return success_response(str(result))

        elif name == "reset_corpus_state":
            _corpus = None
            _text_analyzer = None
            _csv_analyzer = None
            _ml_analyzer = None
            return success_response("Global corpus state has been reset.")

        elif name == "clear_cache":
            try:
                clear_cache()
                return success_response("Cache has been cleared successfully.")
            except Exception as e:
                return error_response(f"Error clearing cache: {str(e)}")

        elif name == "semantic_search":
            if not _corpus:
                return no_corpus_response()

            try:
                from ..semantic import Semantic

                query = arguments.get("query")
                if not query:
                    return error_response("query is required")

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

                return success_response(response_text)

            except ImportError:
                return error_response(
                    "chromadb is not installed. Install with: pip install chromadb"
                )
            except Exception as e:
                return error_response(f"Error during semantic search: {e}")

        elif name == "find_similar_documents":
            if not _corpus:
                return no_corpus_response()

            try:
                from ..semantic import Semantic

                document_ids = arguments.get("document_ids")
                if not document_ids:
                    return error_response("document_ids is required")

                n_results = arguments.get("n_results", 5)
                threshold = arguments.get("threshold", 0.7)

                semantic_analyzer = Semantic(_corpus)
                similar_doc_ids = semantic_analyzer.get_similar_documents(
                    document_ids=document_ids, n_results=n_results, threshold=threshold
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
                    response_text += (
                        "- Literature reviews: Find additional relevant papers\n"
                    )
                    response_text += "- Qualitative research: Identify documents with similar themes\n"
                    response_text += (
                        "- Content grouping: Group similar documents for analysis\n"
                    )
                else:
                    response_text += "No similar documents found above the threshold.\n"
                    response_text += "Try lowering the threshold or using different reference documents.\n"

                return success_response(response_text)

            except ImportError:
                return error_response(
                    "chromadb is not installed. Install with: pip install chromadb"
                )
            except Exception as e:
                return error_response(f"Error finding similar documents: {e}")

        elif name == "semantic_chunk_search":
            if not _corpus:
                return no_corpus_response()

            try:
                from ..semantic import Semantic

                query = arguments.get("query")
                doc_id = arguments.get("doc_id")

                if not query:
                    return error_response("query is required")
                if not doc_id:
                    return error_response("doc_id is required")

                threshold = arguments.get("threshold", 0.5)
                n_results = arguments.get("n_results", 10)

                semantic_analyzer = Semantic(_corpus)
                chunks = semantic_analyzer.get_similar_chunks(
                    query=query, doc_id=doc_id, threshold=threshold, n_results=n_results
                )

                # Prepare response
                response_text = (
                    f"Semantic chunk search completed for query: '{query}'\n"
                )
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
                    response_text += (
                        "You can adjust the threshold to get more or fewer results.\n"
                    )
                else:
                    response_text += (
                        "No chunks matched the query above the threshold.\n"
                    )
                    response_text += (
                        "Try lowering the threshold or use a different query.\n"
                    )

                return success_response(response_text)

            except ImportError:
                return error_response(
                    "chromadb is not installed. Install with: pip install chromadb"
                )
            except Exception as e:
                return error_response(f"Error during semantic chunk search: {e}")

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
                    return success_response(response_text)
                else:
                    return error_response("No DataFrame created")

            except ImportError:
                return error_response(
                    "chromadb is not installed. Install with: pip install chromadb"
                )
            except Exception as e:
                return error_response(f"Error exporting metadata: {e}")

        elif name == "tdabm_analysis":
            if not _corpus:
                return no_corpus_response()

            try:
                from ..tdabm import Tdabm

                y_variable = arguments.get("y_variable")
                x_variables = arguments.get("x_variables")
                radius = arguments.get("radius", 0.3)

                if not y_variable or not x_variables:
                    return error_response(
                        "Both y_variable and x_variables are required"
                    )

                tdabm_analyzer = Tdabm(_corpus)
                result = tdabm_analyzer.generate_tdabm(
                    y=y_variable, x_variables=x_variables, radius=radius, mcp=True
                )

                return success_response(
                    f"TDABM Analysis Complete\n\n{result}\n\n"
                    "Hint: Results are stored in corpus metadata['tdabm']\n"
                    "Hint: Use save_corpus to persist the results\n"
                    "Hint: Visualize with draw_tdabm or use vizcli --tdabm"
                )

            except ValueError as e:
                return error_response(
                    f"Validation Error: {e}\n\n"
                    "Tips:\n"
                    "- Ensure corpus has a DataFrame\n"
                    "- Y variable must be continuous (not binary)\n"
                    "- X variables must be numeric/ordinal\n"
                    "- All variables must exist in the DataFrame"
                )
            except Exception as e:
                return error_response(f"Error during TDABM analysis: {e}")

        # Temporal Analysis Tools
        elif name == "temporal_link_by_time":
            if not _corpus:
                return no_corpus_response()

            try:
                from datetime import timedelta

                from ..temporal import TemporalAnalyzer

                method = arguments.get("method", "nearest")
                time_column = arguments.get("time_column", "timestamp")
                analyzer = TemporalAnalyzer(_corpus)

                if method == "nearest":
                    _corpus = analyzer.link_by_nearest_time(time_column=time_column)
                    return success_response(
                        "Documents linked to nearest dataframe rows by time"
                    )

                elif method == "window":
                    window_seconds = arguments.get("window_seconds", 300)
                    window = timedelta(seconds=window_seconds)
                    _corpus = analyzer.link_by_time_window(
                        time_column=time_column,
                        window_before=window,
                        window_after=window,
                    )
                    return success_response(
                        f"Documents linked within ±{window_seconds}s time window"
                    )

                elif method == "sequence":
                    period = arguments.get("period", "W")
                    _corpus = analyzer.link_by_sequence(
                        time_column=time_column, period=period
                    )
                    return success_response(f"Documents linked in {period} sequences")

                else:
                    return error_response(f"Unknown method: {method}")

            except Exception as e:
                return error_response(f"Error in temporal linking: {e}")

        elif name == "temporal_filter":
            if not _corpus:
                return no_corpus_response()

            try:
                from ..temporal import TemporalAnalyzer

                start_time = arguments.get("start_time")
                end_time = arguments.get("end_time")
                time_column = arguments.get("time_column", "timestamp")

                analyzer = TemporalAnalyzer(_corpus)
                _corpus = analyzer.filter_by_time_range(
                    start_time=start_time, end_time=end_time, time_column=time_column
                )

                doc_count = len(_corpus.documents)
                df_count = len(_corpus.df) if _corpus.df is not None else 0
                return success_response(
                    f"Corpus filtered by time range: {doc_count} documents, {df_count} dataframe rows"
                )

            except Exception as e:
                return error_response(f"Error in temporal filtering: {e}")

        elif name == "temporal_summary":
            if not _corpus:
                return no_corpus_response()

            try:
                from ..temporal import TemporalAnalyzer

                period = arguments.get("period", "W")
                time_column = arguments.get("time_column", "timestamp")

                analyzer = TemporalAnalyzer(_corpus)
                summary = analyzer.get_temporal_summary(
                    time_column=time_column, period=period
                )

                if not summary.empty:
                    response_text = f"Temporal Summary ({period} periods):\n\n"
                    response_text += summary.to_string()
                    return success_response(response_text)
                else:
                    return error_response("No temporal data available for summary")

            except Exception as e:
                return error_response(f"Error in temporal summary: {e}")

        elif name == "temporal_sentiment_trend":
            if not _corpus:
                return no_corpus_response()

            try:
                from ..temporal import TemporalAnalyzer

                period = arguments.get("period", "W")
                aggregation = arguments.get("aggregation", "mean")

                analyzer = TemporalAnalyzer(_corpus)
                trend = analyzer.get_temporal_sentiment_trend(
                    period=period, aggregation=aggregation
                )

                if not trend.empty:
                    response_text = f"Temporal Sentiment Trend ({period} periods, {aggregation}):\n\n"
                    response_text += trend.to_string()
                    return success_response(response_text)
                else:
                    return error_response(
                        "No sentiment data available. Run sentiment analysis first."
                    )

            except Exception as e:
                return error_response(f"Error in temporal sentiment: {e}")

        elif name == "temporal_topics":
            if not _corpus:
                return no_corpus_response()

            try:
                from ..temporal import TemporalAnalyzer

                period = arguments.get("period", "W")
                top_n = arguments.get("top_n", 5)

                analyzer = TemporalAnalyzer(_corpus)
                topics = analyzer.get_temporal_topics(period=period, top_n=top_n)

                if topics:
                    response_text = (
                        f"Temporal Topics (top {top_n} per {period} period):\n\n"
                    )
                    for period_key, topic_list in topics.items():
                        response_text += f"{period_key}: {', '.join(topic_list)}\n"
                    return success_response(response_text)
                else:
                    return error_response(
                        "No temporal data available for topic extraction"
                    )

            except Exception as e:
                return error_response(f"Error in temporal topics: {e}")

        # Embedding-based linking tools
        elif name == "embedding_link":
            if not _corpus:
                return no_corpus_response()

            try:
                from ..embedding_linker import EmbeddingLinker

                similarity_metric = arguments.get("similarity_metric", "cosine")
                top_k = arguments.get("top_k", 1)
                threshold = arguments.get("threshold")
                numeric_columns_str = arguments.get("numeric_columns")

                numeric_columns = None
                if numeric_columns_str:
                    numeric_columns = [
                        c.strip() for c in numeric_columns_str.split(",")
                    ]

                linker = EmbeddingLinker(
                    _corpus,
                    similarity_metric=similarity_metric,
                    use_simple_embeddings=True,
                )
                _corpus = linker.link_by_embedding_similarity(
                    numeric_columns=numeric_columns, threshold=threshold, top_k=top_k
                )

                stats = linker.get_link_statistics()
                response_text = "Embedding-based linking complete\n\n"
                response_text += f"Linked documents: {stats['linked_documents']}/{stats['total_documents']}\n"
                response_text += f"Total links: {stats['total_links']}\n"
                response_text += f"Average similarity: {stats['avg_similarity']:.3f}\n"
                response_text += f"Similarity metric: {similarity_metric}\n"

                return success_response(response_text)

            except ImportError:
                return error_response(
                    "ChromaDB is not installed. Install with: pip install chromadb"
                )
            except Exception as e:
                return error_response(f"Error in embedding linking: {e}")

        elif name == "embedding_link_stats":
            if not _corpus:
                return no_corpus_response()

            try:
                from ..embedding_linker import EmbeddingLinker

                # Check if corpus has embedding links
                has_links = any(
                    "embedding_links" in doc.metadata
                    and doc.metadata["embedding_links"]
                    for doc in _corpus.documents
                )

                if not has_links:
                    return error_response(
                        "No embedding links found. Run embedding_link first."
                    )

                linker = EmbeddingLinker(_corpus, use_simple_embeddings=True)
                stats = linker.get_link_statistics()

                response_text = "Embedding Link Statistics:\n\n"
                response_text += f"Total documents: {stats['total_documents']}\n"
                response_text += f"Linked documents: {stats['linked_documents']}\n"
                response_text += f"Total links: {stats['total_links']}\n"
                response_text += f"Average similarity: {stats['avg_similarity']:.3f}\n"
                response_text += f"Min similarity: {stats['min_similarity']:.3f}\n"
                response_text += f"Max similarity: {stats['max_similarity']:.3f}\n"

                return success_response(response_text)

            except Exception as e:
                return error_response(f"Error getting embedding statistics: {e}")

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
                        text="""# CRISP-T Analysis Workflow

Follow these steps to conduct a comprehensive analysis:

## Data Preparation and Exploration

* **Load your data**
   - Use `load_corpus` tool with either `inp` (existing corpus) or `source` (directory/URL)
   - For CSV data with text columns, specify `text_columns` parameter

* **Inspect the data**
   - Use `list_documents` to see all documents
   - Use `get_df_columns` and `get_df_row_count` if you have numeric data
   - Use `get_document` to examine specific documents

* **Link text to numeric data**
   - Use `temporal_link_by_time` if you have timestamps
   - Use `embedding_link` to link based on semantic similarity, if applicable

## Descriptive Analysis

* **Generate coding dictionary for entire corpus**
   - Use `generate_coding_dictionary` with appropriate `num` and `top_n` parameters
   - This reveals categories (verbs), properties (nouns), and dimensions (adjectives)

* **Perform sentiment analysis**
   - Use `sentiment_analysis` to understand emotional tone
   - Set `documents=true` for document-level analysis

* **Basic statistical exploration**
   - Use `get_df_row` to examine specific data points
   - Review column distributions

## Advanced Pattern Discovery

* **Topic modeling**
   - Use `topic_modeling` to discover latent themes for entire corpus (set appropriate `num_topics`)
   - Use `assign_topics` to assign documents to their dominant topics. PERFORM THIS STEP ALWAYS.
   - Use `clear_cache` before `assign_topics` if you change filters
   - Topics generate keywords that can be used to categorize documents

* **Numerical clustering** (if you have numeric data)
   - Use `kmeans_clustering` to segment your data
   - Review cluster profiles to understand groupings

* **Association rules** (if applicable)
   - Use `extract_categories` for text-based associations
   - Use `association_rules` for numeric pattern mining

## Predictive Modeling (if you have an outcome variable)
* **Classification**
   - Use `decision_tree_classification` to get feature importance rankings
   - Use `svm_classification` for robust classification
   - Use `neural_network_classification` for complex patterns

* **Regression analysis**
    - Use `regression_analysis` to understand factor relationships
    - It auto-detects binary outcomes (logistic) vs continuous (linear)
    - Returns coefficients showing strength and direction of relationships

* **Dimensionality reduction**
    - Use `pca_analysis` to reduce feature space

## Validation and Triangulation

* **Cross-modal analysis**
    - Use linkage and aggregation methods in ML tools to combine text and numeric data
    - Experiment with different linkage methods: nearest, window, sequence
    - Experiment with aggregation methods: majority, mean, median
    - With linked data the outcome variable can be in the text or numeric side

* **Create relationships**
    - Use `add_relationship` to link text keywords (from topics) with numeric columns
    - Example: link topic keywords to demographic or outcome variables
    - Use format like: first="text:healthcare", second="num:age_group", relation="correlates"

* **Validate findings**
    - Compare topic assignments with numerical clusters
    - Validate sentiment patterns with outcome variables
    - Use `get_relationships_for_keyword` to explore connections

* **Save your work**
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

### Cross-Modal Linking
- Use `temporal_link_by_time` to align text documents with time-stamped numeric data
- Use `embedding_link` to connect documents with similar semantic content to numeric records
- If id columns exist, use them to directly link text and numeric data
- If keywords exist in both text and numeric data (as columns), use them for linking

### Link Topic Keywords to Variables

After topic modeling:
- Topics generate keywords representing themes
- Use `add_relationship` to link keywords to relevant dataframe columns
- Example: If topic discusses "satisfaction", link to satisfaction score column

### Compare Patterns

- Cross-reference sentiment with numeric outcomes
- Compare topic distributions across demographic groups
- Validate clustering results using both text and numbers

### Use Relationships

- `add_relationship("text:keyword", "num:column", "correlates")`
- `get_relationships_for_keyword` to explore connections
- Document theoretical justifications for relationships

### Validate Findings

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
