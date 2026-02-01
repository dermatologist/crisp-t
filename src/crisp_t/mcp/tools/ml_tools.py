"""
ML Tools for MCP Server

This module contains machine learning tool definitions and handlers.
Requires ML dependencies to be installed.
"""

import logging
from typing import Any, cast

from mcp.types import Tool

logger = logging.getLogger(__name__)

# Try to import ML if available
try:
    from ...ml import ML

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("ML dependencies not available")
    ML = cast(Any, None)


def get_ml_tools() -> list[Tool]:
    """Get list of ML tools if ML dependencies are available."""
    if not ML_AVAILABLE:
        return []

    return [
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


def handle_ml_tool(
    name: str,
    arguments: dict[str, Any],
    corpus: Any,
    text_analyzer: Any,
    csv_analyzer: Any,
    ml_analyzer: Any,
) -> tuple[list[Any], Any, Any] | None:
    """
    Handle ML tool calls.
    
    Args:
        name: Tool name
        arguments: Tool arguments
        corpus: Current corpus
        text_analyzer: Current text analyzer
        csv_analyzer: Current CSV analyzer
        ml_analyzer: Current ML analyzer (may be None, will be created if needed)
        
    Returns:
        Tuple of (response, updated_corpus, updated_ml_analyzer) or None if tool not handled
    """
    from ..utils.responses import error_response, no_csv_analyzer_response, success_response
    
    # Create a local reference for ML analyzer that can be updated
    _ml_analyzer = ml_analyzer
    
    # ML Tools
    if name == "kmeans_clustering":
        if not csv_analyzer:
            return no_csv_analyzer_response(), corpus, _ml_analyzer
        else:
            if "include" in arguments:
                csv_analyzer.comma_separated_include_columns(
                    arguments.get("include") + "," + arguments.get("outcome", "")
                )

        if not ML_AVAILABLE:
            return error_response("ML dependencies not available"), corpus, _ml_analyzer

        csv_analyzer.retain_numeric_columns_only()

        csv_analyzer.drop_na()
        ml = ML(csv=csv_analyzer)
        result = ml.get_kmeans(
            number_of_clusters=arguments.get("num_clusters", 3),
            verbose=False,
            mcp=True,
        )
        return success_response(str(result)), corpus, _ml_analyzer

    elif name == "decision_tree_classification":
        if not csv_analyzer:
            return no_csv_analyzer_response(), corpus, _ml_analyzer
        else:
            if "include" in arguments:
                csv_analyzer.comma_separated_include_columns(
                    arguments.get("include") + "," + arguments.get("outcome", "")
                )

        if not ML_AVAILABLE:
            return error_response("ML dependencies not available"), corpus, _ml_analyzer

        if not _ml_analyzer:
            _ml_analyzer = ML(csv=csv_analyzer)

        result = _ml_analyzer.get_decision_tree_classes(
            y=arguments["outcome"],
            top_n=arguments.get("top_n", 10),
            mcp=True,
            linkage_method=arguments.get("linkage_method"),
            aggregation=arguments.get("aggregation", "majority"),
        )
        return success_response(str(result)), corpus, _ml_analyzer

    elif name == "svm_classification":
        if not csv_analyzer:
            return no_csv_analyzer_response(), corpus, _ml_analyzer
        else:
            if "include" in arguments:
                csv_analyzer.comma_separated_include_columns(
                    arguments.get("include") + "," + arguments.get("outcome", "")
                )

        if not ML_AVAILABLE:
            return error_response("ML dependencies not available"), corpus, _ml_analyzer

        if not _ml_analyzer:
            _ml_analyzer = ML(csv=csv_analyzer)

        linkage_method = arguments.get("linkage_method")
        aggregation = arguments.get("aggregation", "majority")

        result = _ml_analyzer.svm_confusion_matrix(
            y=arguments["outcome"],
            test_size=0.25,
            mcp=True,
            linkage_method=linkage_method,
            aggregation=aggregation,
        )
        return success_response(str(result)), corpus, _ml_analyzer

    elif name == "neural_network_classification":
        if not csv_analyzer:
            return no_csv_analyzer_response(), corpus, _ml_analyzer
        else:
            if "include" in arguments:
                csv_analyzer.comma_separated_include_columns(
                    arguments.get("include") + "," + arguments.get("outcome", "")
                )

        if not ML_AVAILABLE:
            return error_response("ML dependencies not available"), corpus, _ml_analyzer

        if not _ml_analyzer:
            _ml_analyzer = ML(csv=csv_analyzer)

        linkage_method = arguments.get("linkage_method")
        aggregation = arguments.get("aggregation", "majority")

        result = _ml_analyzer.get_nnet_predictions(
            y=arguments["outcome"],
            mcp=True,
            linkage_method=linkage_method,
            aggregation=aggregation,
        )
        return success_response(str(result)), corpus, _ml_analyzer

    elif name == "regression_analysis":
        if not csv_analyzer:
            return no_csv_analyzer_response(), corpus, _ml_analyzer
        else:
            if "include" in arguments:
                csv_analyzer.comma_separated_include_columns(
                    arguments.get("include") + "," + arguments.get("outcome", "")
                )

        if not ML_AVAILABLE:
            return error_response("ML dependencies not available"), corpus, _ml_analyzer

        if not _ml_analyzer:
            _ml_analyzer = ML(csv=csv_analyzer)

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
        return success_response(str(result)), corpus, _ml_analyzer

    elif name == "pca_analysis":
        if not csv_analyzer:
            return no_csv_analyzer_response(), corpus, _ml_analyzer
        else:
            if "include" in arguments:
                csv_analyzer.comma_separated_include_columns(
                    arguments.get("include") + "," + arguments.get("outcome", "")
                )

        if not ML_AVAILABLE:
            return error_response("ML dependencies not available"), corpus, _ml_analyzer

        if not _ml_analyzer:
            _ml_analyzer = ML(csv=csv_analyzer)

        result = _ml_analyzer.get_pca(
            y=arguments["outcome"],
            n=arguments.get("n_components", 3),
            mcp=True,
            linkage_method=arguments.get("linkage_method"),
            aggregation=arguments.get("aggregation", "majority"),
        )
        return success_response(str(result)), corpus, _ml_analyzer

    elif name == "association_rules":
        if not csv_analyzer:
            return no_csv_analyzer_response(), corpus, _ml_analyzer
        else:
            if "include" in arguments:
                csv_analyzer.comma_separated_include_columns(
                    arguments.get("include") + "," + arguments.get("outcome", "")
                )

        if not ML_AVAILABLE:
            return error_response("ML dependencies not available"), corpus, _ml_analyzer

        if not _ml_analyzer:
            _ml_analyzer = ML(csv=csv_analyzer)

        min_support = arguments.get("min_support", 50) / 100
        min_threshold = arguments.get("min_threshold", 50) / 100

        result = _ml_analyzer.get_apriori(
            y=arguments["outcome"],
            min_support=min_support,
            min_threshold=min_threshold,
            mcp=True,
        )
        return success_response(str(result)), corpus, _ml_analyzer

    elif name == "knn_search":
        if not csv_analyzer:
            return no_csv_analyzer_response(), corpus, _ml_analyzer
        else:
            if "include" in arguments:
                csv_analyzer.comma_separated_include_columns(
                    arguments.get("include") + "," + arguments.get("outcome", "")
                )

        if not ML_AVAILABLE:
            return error_response("ML dependencies not available"), corpus, _ml_analyzer

        if not _ml_analyzer:
            _ml_analyzer = ML(csv=csv_analyzer)

        result = _ml_analyzer.knn_search(
            y=arguments["outcome"],
            n=arguments.get("n", 3),
            r=arguments.get("record", 1),
            mcp=True,
            linkage_method=arguments.get("linkage_method"),
            aggregation=arguments.get("aggregation", "majority"),
        )
        return success_response(str(result)), corpus, _ml_analyzer

    elif name == "lstm_text_classification":
        if not csv_analyzer:
            return no_csv_analyzer_response(), corpus, _ml_analyzer

        if not ML_AVAILABLE:
            return error_response("ML dependencies not available"), corpus, _ml_analyzer

        if not _ml_analyzer:
            _ml_analyzer = ML(csv=csv_analyzer)

        result = _ml_analyzer.get_lstm_predictions(y=arguments["outcome"], mcp=True)
        return success_response(str(result)), corpus, _ml_analyzer
    
    # If we got here, the tool name didn't match
    return None
