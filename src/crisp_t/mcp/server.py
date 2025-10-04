"""
MCP Server for CRISP-T

This module provides an MCP (Model Context Protocol) server that exposes
CRISP-T's text analysis, ML analysis, and corpus manipulation capabilities
as tools, resources, and prompts.
"""

import json
import logging
from typing import Any, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    TextContent,
    Tool,
    Prompt,
    GetPromptResult,
    PromptMessage,
)

from ..helpers.initializer import initialize_corpus
from ..helpers.analyzer import get_csv_analyzer, get_text_analyzer
from ..read_data import ReadData
from ..text import Text
from ..sentiment import Sentiment
from ..cluster import Cluster
from ..csv import Csv

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

# Global state for the server
_corpus = None
_text_analyzer = None
_csv_analyzer = None
_ml_analyzer = None


def _init_corpus(inp: Optional[str] = None, source: Optional[str] = None, text_columns: str = "", ignore_words: str = ""):
    """Initialize corpus from input path or source."""
    global _corpus, _text_analyzer, _csv_analyzer
    
    try:
        _corpus = initialize_corpus(
            source=source,
            inp=inp,
            comma_separated_text_columns=text_columns,
            comma_separated_ignore_words=ignore_words if ignore_words else None,
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
                    uri=f"corpus://document/{doc.id}",
                    name=f"Document: {doc.name or doc.id}",
                    description=doc.description or f"Text content of document {doc.id}",
                    mimeType="text/plain",
                )
            )
    
    return resources


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Read a corpus document by URI."""
    if not uri.startswith("corpus://document/"):
        raise ValueError(f"Unknown resource URI: {uri}")
    
    doc_id = uri.replace("corpus://document/", "")
    
    if not _corpus:
        raise ValueError("No corpus loaded. Use load_corpus tool first.")
    
    doc = _corpus.get_document_by_id(doc_id)
    if not doc:
        raise ValueError(f"Document not found: {doc_id}")
    
    return doc.text


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    tools = [
        # Corpus management tools
        Tool(
            name="load_corpus",
            description="Load a corpus from a folder containing corpus.json or from a source directory/URL. This is the first step in most analyses.",
            inputSchema={
                "type": "object",
                "properties": {
                    "inp": {
                        "type": "string",
                        "description": "Path to folder containing corpus.json"
                    },
                    "source": {
                        "type": "string",
                        "description": "Source directory or URL to read data from"
                    },
                    "text_columns": {
                        "type": "string",
                        "description": "Comma-separated text column names (for CSV data)"
                    },
                    "ignore_words": {
                        "type": "string",
                        "description": "Comma-separated words to ignore during analysis"
                    }
                }
            }
        ),
        Tool(
            name="save_corpus",
            description="Save the current corpus to a folder as corpus.json",
            inputSchema={
                "type": "object",
                "properties": {
                    "out": {
                        "type": "string",
                        "description": "Output folder path to save corpus"
                    }
                },
                "required": ["out"]
            }
        ),
        Tool(
            name="add_document",
            description="Add a new document to the corpus",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "Unique document ID"},
                    "text": {"type": "string", "description": "Document text content"},
                    "name": {"type": "string", "description": "Optional document name"}
                },
                "required": ["doc_id", "text"]
            }
        ),
        Tool(
            name="remove_document",
            description="Remove a document from the corpus by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "Document ID to remove"}
                },
                "required": ["doc_id"]
            }
        ),
        Tool(
            name="get_document",
            description="Get a document by ID from the corpus",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "Document ID"}
                },
                "required": ["doc_id"]
            }
        ),
        Tool(
            name="list_documents",
            description="List all document IDs in the corpus",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="add_relationship",
            description="Add a relationship between text keywords and numeric columns. Used to link topic modeling results with dataframe columns.",
            inputSchema={
                "type": "object",
                "properties": {
                    "first": {"type": "string", "description": "First entity (e.g., 'text:keyword')"},
                    "second": {"type": "string", "description": "Second entity (e.g., 'num:column')"},
                    "relation": {"type": "string", "description": "Relationship type (e.g., 'correlates')"}
                },
                "required": ["first", "second", "relation"]
            }
        ),
        Tool(
            name="get_relationships",
            description="Get all relationships in the corpus",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="get_relationships_for_keyword",
            description="Get relationships involving a specific keyword",
            inputSchema={
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "Keyword to search for"}
                },
                "required": ["keyword"]
            }
        ),
        
        # NLP/Text Analysis Tools
        Tool(
            name="generate_coding_dictionary",
            description="Generate a qualitative coding dictionary with categories (verbs), properties (nouns), and dimensions (adjectives/adverbs). Useful for understanding the main themes and concepts in the corpus.",
            inputSchema={
                "type": "object",
                "properties": {
                    "num": {"type": "integer", "description": "Number of categories to extract", "default": 3},
                    "top_n": {"type": "integer", "description": "Top N items per category", "default": 3}
                }
            }
        ),
        Tool(
            name="topic_modeling",
            description="Perform LDA topic modeling to discover latent topics in the corpus. Returns topics with their associated keywords and weights, useful for categorizing documents by theme.",
            inputSchema={
                "type": "object",
                "properties": {
                    "num_topics": {"type": "integer", "description": "Number of topics to generate", "default": 3},
                    "num_words": {"type": "integer", "description": "Number of words per topic", "default": 5}
                }
            }
        ),
        Tool(
            name="assign_topics",
            description="Assign documents to their dominant topics with contribution percentages. These topic assignments can be used as keywords to filter or categorize documents.",
            inputSchema={
                "type": "object",
                "properties": {
                    "num_topics": {"type": "integer", "description": "Number of topics (should match topic_modeling)", "default": 3}
                }
            }
        ),
        Tool(
            name="extract_categories",
            description="Extract common categories/concepts from the corpus as bag-of-terms with weights",
            inputSchema={
                "type": "object",
                "properties": {
                    "num": {"type": "integer", "description": "Number of categories", "default": 10}
                }
            }
        ),
        Tool(
            name="generate_summary",
            description="Generate an extractive text summary of the entire corpus",
            inputSchema={
                "type": "object",
                "properties": {
                    "weight": {"type": "integer", "description": "Summary weight/length parameter", "default": 10}
                }
            }
        ),
        Tool(
            name="sentiment_analysis",
            description="Perform VADER sentiment analysis on the corpus, providing positive, negative, neutral, and compound scores",
            inputSchema={
                "type": "object",
                "properties": {
                    "documents": {"type": "boolean", "description": "Analyze at document level", "default": False},
                    "verbose": {"type": "boolean", "description": "Verbose output", "default": True}
                }
            }
        ),
        
        # DataFrame/CSV Tools
        Tool(
            name="get_df_columns",
            description="Get all column names from the DataFrame",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="get_df_row_count",
            description="Get the number of rows in the DataFrame",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="get_df_row",
            description="Get a specific row from the DataFrame by index",
            inputSchema={
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "description": "Row index"}
                },
                "required": ["index"]
            }
        ),
    ]
    
    # Add ML tools if available
    if ML_AVAILABLE:
        tools.extend([
            Tool(
                name="kmeans_clustering",
                description="Perform K-Means clustering on numeric data. Useful for segmenting data into groups based on similarity.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "num_clusters": {"type": "integer", "description": "Number of clusters", "default": 3},
                        "outcome": {"type": "string", "description": "Optional outcome variable to exclude"}
                    }
                }
            ),
            Tool(
                name="decision_tree_classification",
                description="Train a decision tree classifier and return variable importance rankings. Shows which features are most predictive of the outcome.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "outcome": {"type": "string", "description": "Target/outcome variable"},
                        "top_n": {"type": "integer", "description": "Top N important features", "default": 10}
                    },
                    "required": ["outcome"]
                }
            ),
            Tool(
                name="svm_classification",
                description="Perform SVM classification and return confusion matrix",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "outcome": {"type": "string", "description": "Target/outcome variable"}
                    },
                    "required": ["outcome"]
                }
            ),
            Tool(
                name="neural_network_classification",
                description="Train a neural network classifier and return predictions with accuracy",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "outcome": {"type": "string", "description": "Target/outcome variable"}
                    },
                    "required": ["outcome"]
                }
            ),
            Tool(
                name="regression_analysis",
                description="Perform linear or logistic regression (auto-detects based on outcome). Returns coefficients for each factor/predictor, showing their relationship with the outcome variable.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "outcome": {"type": "string", "description": "Target/outcome variable"}
                    },
                    "required": ["outcome"]
                }
            ),
            Tool(
                name="pca_analysis",
                description="Perform Principal Component Analysis for dimensionality reduction",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "outcome": {"type": "string", "description": "Variable to exclude from PCA"},
                        "n_components": {"type": "integer", "description": "Number of components", "default": 3}
                    },
                    "required": ["outcome"]
                }
            ),
            Tool(
                name="association_rules",
                description="Generate association rules using Apriori algorithm",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "outcome": {"type": "string", "description": "Variable to exclude"},
                        "min_support": {"type": "integer", "description": "Min support (1-99)", "default": 50},
                        "min_threshold": {"type": "integer", "description": "Min threshold (1-99)", "default": 50}
                    },
                    "required": ["outcome"]
                }
            ),
            Tool(
                name="knn_search",
                description="Find K-nearest neighbors for a specific record",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "outcome": {"type": "string", "description": "Target variable"},
                        "n": {"type": "integer", "description": "Number of neighbors", "default": 3},
                        "record": {"type": "integer", "description": "Record index (1-based)", "default": 1}
                    },
                    "required": ["outcome"]
                }
            ),
        ])
    
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
                return [TextContent(
                    type="text",
                    text=f"Corpus loaded successfully with {doc_count} document(s)"
                )]
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
                metadata={}
            )
            _corpus.add_document(doc)
            return [TextContent(type="text", text=f"Document {arguments['doc_id']} added")]
        
        elif name == "remove_document":
            if not _corpus:
                return [TextContent(type="text", text="No corpus loaded")]
            
            _corpus.remove_document_by_id(arguments["doc_id"])
            return [TextContent(type="text", text=f"Document {arguments['doc_id']} removed")]
        
        elif name == "get_document":
            if not _corpus:
                return [TextContent(type="text", text="No corpus loaded")]
            
            doc = _corpus.get_document_by_id(arguments["doc_id"])
            if doc:
                return [TextContent(type="text", text=json.dumps(doc.model_dump(), indent=2, default=str))]
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
                arguments["first"],
                arguments["second"],
                arguments["relation"]
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
                num=arguments.get("num", 3),
                top_n=arguments.get("top_n", 3)
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
        elif name == "topic_modeling":
            if not _corpus:
                return [TextContent(type="text", text="No corpus loaded")]
            
            cluster = Cluster(corpus=_corpus)
            cluster.build_lda_model(topics=arguments.get("num_topics", 3))
            result = cluster.print_topics(num_words=arguments.get("num_words", 5))
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
        elif name == "assign_topics":
            if not _corpus:
                return [TextContent(type="text", text="No corpus loaded")]
            
            cluster = Cluster(corpus=_corpus)
            cluster.build_lda_model(topics=arguments.get("num_topics", 3))
            result = cluster.format_topics_sentences(visualize=False)
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
        elif name == "extract_categories":
            if not _text_analyzer:
                return [TextContent(type="text", text="No corpus loaded")]
            
            _text_analyzer.make_spacy_doc()
            result = _text_analyzer.print_categories(num=arguments.get("num", 10))
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
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
                verbose=arguments.get("verbose", True)
            )
            return [TextContent(type="text", text=str(result))]
        
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
                return [TextContent(type="text", text=json.dumps(row.to_dict(), indent=2, default=str))]
            return [TextContent(type="text", text="Row not found")]
        
        # ML Tools
        elif name == "kmeans_clustering":
            if not _csv_analyzer:
                return [TextContent(type="text", text="No CSV data available")]
            
            if not ML_AVAILABLE:
                return [TextContent(type="text", text="ML dependencies not available")]
            
            _csv_analyzer.retain_numeric_columns_only()
            _csv_analyzer.drop_na()
            ml = ML(csv=_csv_analyzer)
            clusters, members = ml.get_kmeans(
                number_of_clusters=arguments.get("num_clusters", 3),
                verbose=False
            )
            return [TextContent(type="text", text=json.dumps({
                "clusters": clusters,
                "members": members
            }, indent=2, default=str))]
        
        elif name == "decision_tree_classification":
            if not _csv_analyzer:
                return [TextContent(type="text", text="No CSV data available")]
            
            if not ML_AVAILABLE:
                return [TextContent(type="text", text="ML dependencies not available")]
            
            if not _ml_analyzer:
                _ml_analyzer = ML(csv=_csv_analyzer)
            
            cm, importance = _ml_analyzer.get_decision_tree_classes(
                y=arguments["outcome"],
                top_n=arguments.get("top_n", 10)
            )
            return [TextContent(type="text", text=json.dumps({
                "confusion_matrix": cm,
                "feature_importance": importance
            }, indent=2, default=str))]
        
        elif name == "svm_classification":
            if not _csv_analyzer:
                return [TextContent(type="text", text="No CSV data available")]
            
            if not ML_AVAILABLE:
                return [TextContent(type="text", text="ML dependencies not available")]
            
            if not _ml_analyzer:
                _ml_analyzer = ML(csv=_csv_analyzer)
            
            result = _ml_analyzer.svm_confusion_matrix(
                y=arguments["outcome"],
                test_size=0.25
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
        elif name == "neural_network_classification":
            if not _csv_analyzer:
                return [TextContent(type="text", text="No CSV data available")]
            
            if not ML_AVAILABLE:
                return [TextContent(type="text", text="ML dependencies not available")]
            
            if not _ml_analyzer:
                _ml_analyzer = ML(csv=_csv_analyzer)
            
            result = _ml_analyzer.get_nnet_predictions(y=arguments["outcome"])
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
        elif name == "regression_analysis":
            if not _csv_analyzer:
                return [TextContent(type="text", text="No CSV data available")]
            
            if not ML_AVAILABLE:
                return [TextContent(type="text", text="ML dependencies not available")]
            
            if not _ml_analyzer:
                _ml_analyzer = ML(csv=_csv_analyzer)
            
            result = _ml_analyzer.get_regression(y=arguments["outcome"])
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
        elif name == "pca_analysis":
            if not _csv_analyzer:
                return [TextContent(type="text", text="No CSV data available")]
            
            if not ML_AVAILABLE:
                return [TextContent(type="text", text="ML dependencies not available")]
            
            if not _ml_analyzer:
                _ml_analyzer = ML(csv=_csv_analyzer)
            
            result = _ml_analyzer.get_pca(
                y=arguments["outcome"],
                n=arguments.get("n_components", 3)
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
        elif name == "association_rules":
            if not _csv_analyzer:
                return [TextContent(type="text", text="No CSV data available")]
            
            if not ML_AVAILABLE:
                return [TextContent(type="text", text="ML dependencies not available")]
            
            if not _ml_analyzer:
                _ml_analyzer = ML(csv=_csv_analyzer)
            
            min_support = arguments.get("min_support", 50) / 100
            min_threshold = arguments.get("min_threshold", 50) / 100
            
            result = _ml_analyzer.get_apriori(
                y=arguments["outcome"],
                min_support=min_support,
                min_threshold=min_threshold
            )
            return [TextContent(type="text", text=str(result))]
        
        elif name == "knn_search":
            if not _csv_analyzer:
                return [TextContent(type="text", text="No CSV data available")]
            
            if not ML_AVAILABLE:
                return [TextContent(type="text", text="ML dependencies not available")]
            
            if not _ml_analyzer:
                _ml_analyzer = ML(csv=_csv_analyzer)
            
            result = _ml_analyzer.knn_search(
                y=arguments["outcome"],
                n=arguments.get("n", 3),
                r=arguments.get("record", 1)
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
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
            arguments=[]
        ),
        Prompt(
            name="triangulation_guide",
            description="Guide for triangulating qualitative and quantitative findings",
            arguments=[]
        ),
    ]


@app.get_prompt()
async def get_prompt(name: str, arguments: dict[str, str] | None = None) -> GetPromptResult:
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
"""
                    )
                )
            ]
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

Triangulation involves validating findings by comparing and contrasting results from different analytical methods or data sources. In CRISP-T, this means linking textual insights with numerical patterns.

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
"""
                    )
                )
            ]
        )
    
    raise ValueError(f"Unknown prompt: {name}")


async def main():
    """Main entry point for the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )
