# CRISP-T Framework User Instructions

## Overview

CRISP-T (**CRoss** **I**ndustry **S**tandard **P**rocess for **T**riangulation) is a comprehensive framework that integrates textual data (as a list of documents) and numeric data (as Pandas DataFrames) into structured classes that retain metadata from various analytical processes. This framework enables researchers to analyze qualitative and quantitative data using advanced NLP, machine learning, and statistical techniques.

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

## Metadata Captured During Analysis (Work In Progress)

### Document-Level Metadata
- **Document ID**: Unique identifier for each document
- **Processing timestamps**: When document was analyzed
- **Language detection**: Identified language of text
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
- **Inertia scores**: Within-cluster sum of squares
- **Silhouette scores**: Cluster quality metrics

### Machine Learning Metadata
- **Model performance**: Accuracy, precision, recall, F1-scores
- **Cross-validation results**: Model stability metrics
- **Feature importance**: Variable significance rankings
- **Confusion matrices**: Classification error patterns
- **Regression metrics**: MSE, R², coefficients, and intercepts for linear and logistic regression models

### Association Rule Metadata
- **Support values**: Frequency of item combinations
- **Confidence scores**: Rule reliability measures
- **Lift values**: Rule significance indicators

## Using Metadata for Theorizing

### 1. **Triangulation Analysis**
- Compare sentiment patterns with numerical outcomes
- Cross-reference topic assignments with demographic variables
- Validate clustering results across textual and numerical dimensions

### 2. **Theory Development**
- Use association rules to identify unexpected relationships
- Leverage topic coherence to validate theoretical constructs
- Apply cluster profiling to develop typologies

### 3. **Hypothesis Testing**
- Compare model performance across different theoretical frameworks
- Use cross-validation to test the theory's generalizability
- Apply statistical tests to validate discovered patterns

### 4. **Pattern Validation**
- Cross-check findings using multiple analytical approaches
- Verify sentiment-outcome relationships with statistical tests
- Validate topic model outputs with expert domain knowledge

## Recommended Sequence of Analysis

### Phase 1: Data Preparation and Exploration
1. **Load and inspect data**
   ```python
   # Load textual data
   corpus = read_data.create_corpus(name="Study", description="Analysis")
   text = Text(corpus=corpus)

   # Load numerical data
   csv = Csv()
   csv.read_csv("data.csv", comma_separated_text_columns="text_col1,text_col2")
   ```

2. **Data cleaning**
   ```python
   csv.mark_missing()
   csv.mark_duplicates()
   csv.get_shape()
   csv.get_column_types()
   ```

3. **Initial text exploration**
   ```python
   text.document_count()
   text.common_words(index=20)
   text.common_nouns(index=15)
   ```

### Phase 2: Descriptive Analysis
4. **Generate coding dictionary**
   ```python
   text.print_coding_dictionary(num=15, top_n=10)
   ```

5. **Perform sentiment analysis**
   ```python
   sentiment = Sentiment(corpus=corpus)
   sentiment.get_sentiment(documents=True, verbose=True)
   ```
   
### Phase 3: Advanced Pattern Discovery
6. **Topic modelling**
   ```python
   cluster = Cluster(corpus=corpus)
   cluster.build_lda_model()
   cluster.print_topics(num_words=10, verbose=True)
   cluster.format_topics_sentences(visualize=True)
   ```

7. **Numerical clustering**
   ```python
   clusters, members = ml.get_kmeans(number_of_clusters=5, verbose=True)
   ml.profile(members, number_of_clusters=5)
   ```

8. **Association rule mining**
   ```python
   text.category_association(num=15)
   ml.get_apriori(y="target", min_support=0.7, min_threshold=2)
   ```

### Phase 4: Predictive Modeling
10. **Machine learning classification**
    ```python
    ml.get_xgb_classes(y="target", oversample=True)
    ml.svm_confusion_matrix(y="target", test_size=0.3)
    ml.get_nnet_predictions(y="target")
    ```

11. **Dimensionality reduction**
    ```python
    ml.get_pca(y="target", n=5)
    ```

### Phase 5: Validation and Triangulation
12. **Cross-validation of findings**
    - Compare topic assignments with numerical clusters
    - Validate sentiment patterns with outcome variables
    - Test association rules across different data subsets

13. **Theory testing**
    - Apply different theoretical lenses to the same data
    - Compare model performance across theoretical frameworks
    - Validate discovered patterns with external data

14. **Report generation**
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
