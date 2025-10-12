# CRISP-T Framework User Instructions

## Overview

CRISP-T (**CRoss** **I**ndustry **S**tandard **P**rocess for **T**riangulation) is a framework that integrates textual data (as a list of documents) and numeric data (as Pandas DataFrames) into structured classes that retain metadata from various analytical processes. This framework enables researchers to analyze qualitative and quantitative data using advanced NLP, machine learning, and statistical techniques. This is under active development; please [report any issues or feature requests on GitHub](https://github.com/dermatologist/crisp-t/issues).


## Metadata Captured During Analysis (Work In Progress)

### Document-Level Metadata
- **Document ID**: Unique identifier for each document
- **Token counts**: Number of words, sentences, paragraphs
- **Sentiment scores**: Compound, positive, negative, neutral scores

### Topic Modeling Metadata
- **Topic assignments**: Dominant topic per document with probability
- **Topic coherence**: Model quality metrics
- **Perplexity scores**: Model performance indicators
- **Topic word distributions**: Most probable words per topic

### Clustering Metadata
- **Cluster assignments**: Document-to-cluster mappings
- **Cluster centroids**: Central points of each cluster
- **Inertia scores**: Within-cluster sum of squares (WIP)
- **Silhouette scores**: Cluster quality metrics (WIP)

### Machine Learning Metadata
- **Model performance**: Accuracy, precision, recall, F1-scores
- **Cross-validation results**: Model stability metrics
- **Feature importance**: Variable significance rankings
- **Confusion matrices**: Classification error patterns
- **Regression metrics**: MSE, R², coefficients, and intercepts for linear and logistic regression models

### Association Rule Metadata
- **Support values**: Frequency of item combinations
- **Confidence scores**: Rule reliability measures
- **Lift values**: Rule significance indicators (WIP)

## Using Metadata for Theorizing

### Triangulation Analysis

#### Semantic Search Integration
- Use `get_similar` to retrieve documents relevant to a specific query.
- Combine semantic search results with topic modeling and sentiment analysis to validate textual patterns.

#### Metadata Export for Validation
- Use `get_df` to export metadata from semantic search or topic modeling.
- Add metadata columns (e.g., sentiment scores, topic assignments) to the DataFrame for numerical analysis.
- Validate relationships between textual insights and numeric patterns using regression or clustering.

###  Theory Development

#### Hypothesis Generation
- Link retrieved documents to numeric variables using `add_relationship`.

#### Metadata for Pattern Validation
- Export metadata to DataFrame and analyze correlations between text-based themes and numeric outcomes.
- Use association rules and clustering to discover unexpected relationships.

### Example Workflow
1. Perform semantic search to retrieve documents related to "customer satisfaction."
2. Export metadata (e.g., sentiment scores, topic assignments) to DataFrame.
3. Use regression analysis to test the impact of sentiment on satisfaction scores.
4. Validate findings using clustering and association rules.
5. Document relationships and theoretical implications.

## Recommended Sequence of Analysis

### Phase 1: Data Preparation and Exploration
### Phase 2: Descriptive Analysis
### Phase 3: Advanced Pattern Discovery
### Phase 4: Predictive Modeling
### Phase 5: Validation and Triangulation
- **Cross-validation of findings**
    - Compare topic assignments with numerical clusters
    - Validate sentiment patterns with outcome variables
    - Test association rules across different data subsets

- **Theory testing**
    - Apply different theoretical lenses to the same data
    - Compare model performance across theoretical frameworks
    - Validate discovered patterns with external data

- **Report generation**
    - Compile metadata from all analyses
    - Create visualizations of key findings
    - Document theoretical implications

### Quality Assurance Checklist
- [ ] Data cleaning (missing values, duplicates handled)
- [ ] Multiple analytical approaches applied to the same research question
- [ ] Model performance metrics documented
- [ ] Statistical significance of findings verified
- [ ] Theoretical coherence of results evaluated
- [ ] Findings triangulated across textual and numerical analyses
- [ ] Metadata preserved for reproducibility
- [ ] Results validated with domain expertise

This systematic approach ensures comprehensive analysis while maintaining theoretical rigor and methodological transparency.


## Command Line Scripts (Quick Reference)

CRISP-T now provides three main command-line scripts:

- `crisp` — Main CLI for triangulation and analysis
- `crispviz` — Visualization CLI for corpus data
- `crispt` — Corpus manipulation CLI
- `crisp-mcp` -- MCP Server for agentic AI

### crisp (Analytical CLI)
- Use `--source PATH|URL` to ingest from a directory (reads .txt and .pdf) or URL. Use `--sources` multiple times to ingest from several locations.
- Use `--inp PATH` to load an existing corpus from a folder containing `corpus.json` (and optional `corpus_df.csv`).
- Use `--out PATH` to save the corpus to a folder (as `corpus.json`) or to act as a base path for analysis outputs (e.g., `results_topics.json`).
- Use `--filters key=value` (repeatable) to retain only documents with matching metadata values; invalid formats raise an error.

### crispviz (Visualization CLI)
- Use `--inp`, `--source`, or `--sources` to specify input corpus or sources
- Use `--out` to specify output directory for PNG images
- Visualization flags: `--freq`, `--by-topic`, `--wordcloud`, `--top-terms`, `--corr-heatmap`

### crispt (Corpus Manipulation CLI)
- Use `--id`, `--name`, `--description` to set corpus metadata
- Use `--doc` to add documents (`id|name|text` or `id|text`)
- Use `--remove-doc` to remove documents by ID
- Use `--meta` to add/update corpus metadata
- Use `--add-rel` to add relationships
- Use `--clear-rel` to clear all relationships
- Use `--out` to save corpus to folder/file as `corpus.json`
- Use `--inp` to load corpus from folder/file containing `corpus.json`
- Query options: `--df-cols`, `--df-row-count`, `--df-row INDEX`, `--doc-ids`, `--doc-id ID`, `--relationships`, `--relationships-for-keyword KEYWORD`

## Available Functions for Textual Data Analysis

### Core Text Processing (`Text` class)

#### Document Management
- `document_count()` - Returns count of documents in corpus
- `filter_documents(metadata_key, metadata_value)` - Filters documents by metadata

#### Lexical Analysis
- `dimensions(word, index=3)` - Analyzes word dimensions and contexts
- `attributes(word, index=3)` - Extracts word attributes and properties
- `generate_summary(weight=10)` - Generates extractive text summaries

#### Coding and Categorization
- `print_coding_dictionary(num=10, top_n=5)` - Creates qualitative coding dictionary
- `print_categories(spacy_doc=None, num=10)` - Extracts document categories
- `category_basket(num=10)` - Creates category baskets for analysis
- `category_association(num=10)` - Performs association rule mining on categories

### Topic Modelling and Clustering (`Cluster` class)

#### Document Analysis
- `most_representative_docs()` - Finds documents most representative of each topic
- `topics_per_document(start=0, end=1)` - Shows topic distribution per document
- `vectorizer(docs, titles, num_clusters=4, visualize=False)` - Document vectorization and clustering

### Sentiment Analysis (`Sentiment` class)

- `get_sentiment(documents=False, verbose=True)` - Performs VADER sentiment analysis
- `max_sentiment(score)` - Identifies documents with maximum sentiment scores

## Available Functions for Numeric Data Analysis

### Data Preprocessing (`Csv` class)

#### Data Quality
- `mark_missing()` - Identifies missing values
- `mark_duplicates()` - Flags duplicate records
- `drop_na()` - Removes rows with missing values
- `restore_df()` - Restores original DataFrame

#### Feature Engineering
- `oversample()` - Applies oversampling for imbalanced data
- `one_hot_encode_strings_in_df()` - Encodes categorical variables

### Machine Learning (`ML` class)

#### Clustering
- `get_kmeans(number_of_clusters=3, seed=42, verbose=True)` - K-Means clustering
- `profile(members, number_of_clusters=3)` - Cluster profiling and analysis

#### Classification
- `get_nnet_predictions(y)` - Neural network predictions
- `svm_confusion_matrix(y, test_size=0.25, random_state=0)` - SVM classification with confusion matrix
- `get_xgb_classes(y, oversample=False, test_size=0.25, random_state=0)` - XGBoost classification

#### Regression
- `get_regression(y)` - Linear or logistic regression (automatically detects binary outcomes for logistic regression vs continuous for linear regression)

#### Pattern Mining
- `get_apriori(y, min_support=0.9, use_colnames=True, min_threshold=3)` - Association rule mining
- `knn_search(y, n=3, r=3)` - K-nearest neighbor search

#### Dimensionality Reduction
- `get_pca(y, n=3)` - Principal Component Analysis

## Semantic Search and Metadata Export

### Semantic Search (`Semantic` class)

#### Document Retrieval
- `get_similar(query, n_results=5)` - Finds documents similar to a query using semantic similarity.
  - **query**: The search query text.
  - **n_results**: Number of similar documents to return (default: 5).

#### Metadata Export
- `get_df(metadata_keys=None)` - Exports ChromaDB collection metadata as a Pandas DataFrame.
  - **metadata_keys**: Comma-separated list of metadata keys to include (optional, includes all if not specified).


