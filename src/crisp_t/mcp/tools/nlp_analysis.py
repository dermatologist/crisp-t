"""
NLP/Text Analysis Tools for MCP Server

This module contains tools for natural language processing and text analysis
including coding dictionaries, topic modeling, categorization, and sentiment analysis.
"""

import json
import logging
from typing import Any

from mcp.types import TextContent, Tool

from ...helpers.cluster import Cluster
from ...helpers.sentiment import Sentiment
from ..utils.responses import (
    no_corpus_response,
    success_response,
)

logger = logging.getLogger(__name__)


def get_nlp_analysis_tools() -> list[Tool]:
    """Return list of NLP/Text Analysis tool definitions."""
    return [
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
    ]


def handle_nlp_analysis_tool(
    name: str,
    arguments: dict[str, Any],
    corpus: Any,
    text_analyzer: Any,
    csv_analyzer: Any,
    ml_analyzer: Any,
) -> tuple[list[TextContent], Any, Any] | None:
    """Handle NLP/Text Analysis tool calls.
    
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
    if name == "generate_coding_dictionary":
        if not text_analyzer:
            return no_corpus_response(), corpus, ml_analyzer

        text_analyzer.make_spacy_doc()
        result = text_analyzer.print_coding_dictionary(
            num=arguments.get("num", 3), top_n=arguments.get("top_n", 3)
        )
        return success_response(json.dumps(result, indent=2, default=str)), corpus, ml_analyzer

    elif name == "topic_modeling":
        if not corpus:
            return no_corpus_response(), corpus, ml_analyzer

        cluster = Cluster(corpus=corpus)
        cluster.build_lda_model(topics=arguments.get("num_topics", 3))
        result = cluster.print_topics(num_words=arguments.get("num_words", 5))
        return success_response(json.dumps(result, indent=2, default=str)), corpus, ml_analyzer

    elif name == "assign_topics":
        if not corpus:
            return no_corpus_response(), corpus, ml_analyzer

        cluster = Cluster(corpus=corpus)
        cluster.build_lda_model(topics=arguments.get("num_topics", 3))
        result = cluster.format_topics_sentences(visualize=False)
        return success_response(json.dumps(result, indent=2, default=str)), corpus, ml_analyzer

    elif name == "extract_categories":
        if not text_analyzer:
            return no_corpus_response(), corpus, ml_analyzer

        text_analyzer.make_spacy_doc()
        result = text_analyzer.print_categories(num=arguments.get("num", 10))
        return success_response(json.dumps(result, indent=2, default=str)), corpus, ml_analyzer

    elif name == "generate_summary":
        if not text_analyzer:
            return no_corpus_response(), corpus, ml_analyzer

        text_analyzer.make_spacy_doc()
        result = text_analyzer.generate_summary(weight=arguments.get("weight", 10))
        return success_response(str(result)), corpus, ml_analyzer

    elif name == "sentiment_analysis":
        if not corpus:
            return no_corpus_response(), corpus, ml_analyzer

        sentiment = Sentiment(corpus=corpus)
        result = sentiment.get_sentiment(
            documents=arguments.get("documents", False),
            verbose=arguments.get("verbose", True),
        )
        return success_response(str(result)), corpus, ml_analyzer

    # Tool not handled by this module
    return None
