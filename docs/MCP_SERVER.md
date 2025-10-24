# CRISP-T MCP Server Guide

## Overview

The CRISP-T MCP (Model Context Protocol) server exposes all CRISP-T functionality through a standardized interface that can be used by AI assistants and other MCP-compatible clients.

## Installation

Install CRISP-T with MCP support:

```bash
pip install crisp-t
```

For machine learning features:

```bash
pip install crisp-t[ml]
```

## Starting the Server

The MCP server runs via stdio and can be started with:

```bash
crisp-mcp
```

Or using Python directly:

```bash
python -m crisp_t.mcp
```

## Client Configuration

### Claude Desktop

Add the following to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "crisp-t": {
      "command": "crisp-mcp"
    }
  }
}
```

After adding the configuration, restart Claude Desktop. The CRISP-T tools will be available in the MCP menu.

### Other MCP Clients

Configure your MCP client to run the `crisp-mcp` command via stdio. Consult your client's documentation for specific configuration instructions.

## Available Tools

### Corpus Management

#### `load_corpus`
Load a corpus from a folder or source directory.

**Arguments:**
- `inp` (optional): Path to folder containing corpus.json
- `source` (optional): Source directory or URL to read data from
- `text_columns` (optional): Comma-separated text column names for CSV data
- `ignore_words` (optional): Comma-separated words to ignore during analysis

**Example:**
```json
{
  "inp": "/path/to/corpus_folder"
}
```

#### `save_corpus`
Save the current corpus to a folder.

**Arguments:**
- `out` (required): Output folder path

#### `add_document`
Add a new document to the corpus.

**Arguments:**
- `doc_id` (required): Unique document ID
- `text` (required): Document text content
- `name` (optional): Document name

#### `remove_document`
Remove a document by ID.

**Arguments:**
- `doc_id` (required): Document ID to remove

#### `get_document`
Get document details by ID.

**Arguments:**
- `doc_id` (required): Document ID

#### `list_documents`
List all document IDs in the corpus.

#### `add_relationship`
Add a relationship between text keywords and numeric columns.

**Arguments:**
- `first` (required): First entity (e.g., "text:healthcare")
- `second` (required): Second entity (e.g., "num:age_group")
- `relation` (required): Relationship type (e.g., "correlates")

**Example:**
```json
{
  "first": "text:satisfaction",
  "second": "num:rating_score",
  "relation": "predicts"
}
```

#### `get_relationships`
Get all relationships in the corpus.

#### `get_relationships_for_keyword`
Get relationships involving a specific keyword.

**Arguments:**
- `keyword` (required): Keyword to search for

### NLP/Text Analysis

#### `generate_coding_dictionary`
Generate a qualitative coding dictionary with categories, properties, and dimensions.

**Arguments:**
- `num` (optional): Number of categories to extract (default: 3)
- `top_n` (optional): Top N items per category (default: 3)

**Returns:**
- Categories (verbs representing main actions/themes)
- Properties (nouns associated with categories)
- Dimensions (adjectives/adverbs describing properties)

#### `topic_modeling`
Perform LDA topic modeling to discover latent topics.

**Arguments:**
- `num_topics` (optional): Number of topics to generate (default: 3)
- `num_words` (optional): Number of words per topic (default: 5)

**Returns:**
Topics with keywords and weights showing word importance within each topic.

#### `assign_topics`
Assign documents to their dominant topics.

**Arguments:**
- `num_topics` (optional): Number of topics (should match topic_modeling, default: 3)

**Returns:**
Document-topic assignments with contribution percentages. These assignments create keyword labels that can be used to filter or categorize documents.

#### `extract_categories`
Extract common categories/concepts from the corpus.

**Arguments:**
- `num` (optional): Number of categories (default: 10)

#### `generate_summary`
Generate an extractive text summary of the corpus.

**Arguments:**
- `weight` (optional): Summary length parameter (default: 10)

#### `sentiment_analysis`
Perform VADER sentiment analysis.

**Arguments:**
- `documents` (optional): Analyze at document level (default: false)
- `verbose` (optional): Verbose output (default: true)

**Returns:**
Sentiment scores (negative, neutral, positive, compound).

### Semantic Search (requires chromadb)

#### `semantic_search`
Find documents similar to a query using semantic similarity.

**Arguments:**
- `query` (required): Search query text
- `n_results` (optional): Number of similar documents to return (default: 5)

**Returns:**
List of similar documents with their IDs and names, ranked by semantic similarity.

**Example:**
```json
{
  "query": "machine learning and AI",
  "n_results": 5
}
```

#### `find_similar_documents`
Find documents similar to a given set of reference documents based on semantic similarity. This tool is particularly useful for **literature reviews** and **qualitative research** where you want to find additional documents that are similar to a set of known relevant documents. It can also be used to identify documents with similar themes, topics, or content for grouping and analysis purposes.

**Arguments:**
- `document_ids` (required): A single document ID or comma-separated list of document IDs to use as reference
- `n_results` (optional): Number of similar documents to return (default: 5)
- `threshold` (optional): Minimum similarity threshold 0-1 (default: 0.7). Only documents above this threshold are returned

**Returns:**
List of document IDs similar to the reference documents, excluding the reference documents themselves.

**Example:**
```json
{
  "document_ids": "doc1,doc5,doc12",
  "n_results": 10,
  "threshold": 0.7
}
```

**Use Cases:**
- Literature reviews: Find papers similar to known relevant papers
- Qualitative research: Identify documents with similar themes
- Content grouping: Group similar documents for analysis
- Document recommendation: Suggest related documents to researchers

#### `semantic_chunk_search`
Perform semantic search on chunks of a specific document. This tool is useful for **coding and annotating documents** by identifying relevant sections that match specific concepts or themes.

**Arguments:**
- `query` (required): Search query text (concept or set of concepts)
- `doc_id` (required): Document ID to search within
- `threshold` (optional): Minimum similarity threshold 0-1 (default: 0.5). Only chunks above this threshold are returned
- `n_results` (optional): Maximum number of chunks to retrieve before filtering (default: 10)

**Returns:**
List of matching text chunks from the specified document that can be used for qualitative analysis or document annotation.

**Example:**
```json
{
  "query": "patient satisfaction",
  "doc_id": "interview_01",
  "threshold": 0.6,
  "n_results": 10
}
```

**Use Cases:**
- Coding qualitative interview transcripts for specific themes
- Identifying sections of documents relevant to research questions
- Annotating documents with concept labels
- Finding evidence for theoretical constructs within texts

#### `export_metadata_df`
Export ChromaDB collection metadata as a pandas DataFrame.

**Arguments:**
- `metadata_keys` (optional): Comma-separated list of metadata keys to include

**Returns:**
DataFrame with document metadata that can be merged with numeric data for mixed-methods analysis.

### TDABM (Topological Data Analysis Ball Mapper)

#### `tdabm_analysis`
Perform Topological Data Analysis Ball Mapper (TDABM) analysis to uncover hidden, global patterns in complex, noisy, or high-dimensional data.

Based on the algorithm by Rudkin and Dlotko (2024), TDABM creates a point cloud from multidimensional data and covers it with overlapping balls, revealing topological structure and relationships between variables.

**Arguments:**
- `y_variable` (required): Name of the continuous Y variable to analyze
- `x_variables` (required): Comma-separated list of ordinal/numeric X variable names
- `radius` (optional): Radius for ball coverage (default: 0.3). Smaller values create more detailed mappings.

**Example:**
```json
{
  "y_variable": "satisfaction",
  "x_variables": "age,income,education",
  "radius": 0.3
}
```

**Returns:**
Analysis results in JSON format including landmark points, their locations, connections, and mean Y values. Results are stored in corpus metadata['tdabm'].

**Use Cases:**
- Discovering hidden patterns in multidimensional data
- Visualizing relationships between multiple variables
- Identifying clusters and connections in complex datasets
- Performing model-free exploratory data analysis
- Understanding global structure in high-dimensional data

**Note:** After running TDABM analysis, use `save_corpus` to persist results, then visualize with `crispviz --tdabm`.

**Reference:**
Rudkin, S., & Dlotko, P. (2024). Topological Data Analysis Ball Mapper for multidimensional data visualization. *Paper reference to be added - algorithm implementation based on the TDABM methodology described by the authors.*

### DataFrame/CSV Operations

#### `get_df_columns`
Get all column names from the DataFrame.

#### `get_df_row_count`
Get the number of rows in the DataFrame.

#### `get_df_row`
Get a specific row by index.

**Arguments:**
- `index` (required): Row index

### Machine Learning (requires crisp-t[ml])

#### `kmeans_clustering`
Perform K-Means clustering on numeric data.

**Arguments:**
- `num_clusters` (optional): Number of clusters (default: 3)
- `outcome` (optional): Outcome variable to exclude

**Returns:**
Cluster assignments and membership information.

#### `decision_tree_classification`
Train a decision tree classifier and return variable importance.

**Arguments:**
- `outcome` (required): Target/outcome variable
- `top_n` (optional): Top N important features (default: 10)

**Returns:**
- Confusion matrix
- Feature importance rankings (shows which variables are most predictive)

#### `svm_classification`
Perform SVM classification.

**Arguments:**
- `outcome` (required): Target/outcome variable

**Returns:**
Confusion matrix showing classification performance.

#### `neural_network_classification`
Train a neural network classifier.

**Arguments:**
- `outcome` (required): Target/outcome variable

**Returns:**
Predictions and accuracy metrics.

#### `regression_analysis`
Perform linear or logistic regression (auto-detects based on outcome type).

**Arguments:**
- `outcome` (required): Target/outcome variable

**Returns:**
- Model type (linear or logistic)
- Coefficients for each factor (showing strength and direction of relationships)
- Intercept
- Performance metrics (R², accuracy, etc.)

#### `pca_analysis`
Perform Principal Component Analysis.

**Arguments:**
- `outcome` (required): Variable to exclude from PCA
- `n_components` (optional): Number of components (default: 3)

#### `association_rules`
Generate association rules using Apriori algorithm.

**Arguments:**
- `outcome` (required): Variable to exclude
- `min_support` (optional): Minimum support 1-99 (default: 50)
- `min_threshold` (optional): Minimum threshold 1-99 (default: 50)

#### `knn_search`
Find K-nearest neighbors for a specific record.

**Arguments:**
- `outcome` (required): Target variable
- `n` (optional): Number of neighbors (default: 3)
- `record` (optional): Record index, 1-based (default: 1)

## Resources

The server exposes corpus documents as resources that can be read:

- `corpus://document/{id}` - Access document text content by ID

## Prompts

### `analysis_workflow`
Complete step-by-step guide for conducting a CRISP-T analysis based on INSTRUCTIONS.md.

This prompt provides:
- Data preparation steps
- Descriptive analysis workflow
- Advanced pattern discovery techniques
- Predictive modeling approaches
- Validation and triangulation strategies

### `triangulation_guide`
Guide for triangulating qualitative and quantitative findings.

This prompt explains:
- How to link topic keywords with numeric variables
- Strategies for comparing patterns across data types
- Using relationships to document connections
- Best practices for validation

## Example Workflow

Here's a typical analysis workflow using the MCP server:

1. **Load data**
   ```
   load_corpus(inp="/path/to/corpus")
   ```

2. **Explore the data**
   ```
   list_documents()
   get_df_columns()
   get_df_row_count()
   ```

3. **Perform text analysis**
   ```
   generate_coding_dictionary(num=10, top_n=5)
   topic_modeling(num_topics=5, num_words=10)
   assign_topics(num_topics=5)
   sentiment_analysis(documents=true)
   ```

4. **Perform numeric analysis**
   ```
   regression_analysis(outcome="satisfaction_score")
   decision_tree_classification(outcome="readmission", top_n=10)
   ```

5. **Link findings**
   ```
   add_relationship(
     first="text:healthcare_access",
     second="num:insurance_status",
     relation="correlates"
   )
   ```

6. **Save results**
   ```
   save_corpus(out="/path/to/output")
   ```

## Key Features

### Topic Modeling Creates Keywords
When you use `topic_modeling` and `assign_topics`, the tool assigns topic keywords to documents. These keywords can then be used to:
- Filter documents by theme
- Create relationships with numeric columns
- Categorize documents for further analysis

### Regression Shows Coefficients
The `regression_analysis` tool returns coefficients for each predictor variable, showing:
- Strength of relationship (larger absolute value = stronger)
- Direction of relationship (positive or negative)
- Statistical significance

### Decision Trees Show Importance
The `decision_tree_classification` tool returns variable importance rankings, indicating which features are most predictive of the outcome.

### Relationships Link Data Types
Use `add_relationship` to document connections between:
- Text findings (keywords from topics)
- Numeric variables (DataFrame columns)
- Theoretical constructs

Example: Link "healthcare_quality" keyword to "patient_satisfaction" column with relation "predicts".

## Tips for Effective Use

1. **Always load corpus first**: Most tools require a loaded corpus to function.

2. **Use prompts for guidance**: Request the `analysis_workflow` or `triangulation_guide` prompts for step-by-step instructions.

3. **Save frequently**: Use `save_corpus` to preserve analysis metadata and relationships.

4. **Document relationships**: Use `add_relationship` to link textual findings with numeric variables based on your theoretical framework.

5. **Iterate and refine**: Analysis is iterative. Load your saved corpus and continue refining based on new insights.

## Troubleshooting

### "No corpus loaded" error
Make sure to call `load_corpus` before using other tools.

### "ML dependencies not available" error
Install the ML extras:
```bash
pip install crisp-t[ml]
```

### Server not appearing in Claude Desktop
1. Check that the configuration file is in the correct location
2. Verify the JSON syntax is valid
3. Restart Claude Desktop after adding the configuration
4. Check that `crisp-mcp` command is in your PATH

## Additional Resources

- [CRISP-T Documentation](https://dermatologist.github.io/crisp-t/)
- [INSTRUCTION.md](/notes/INSTRUCTION.md) - Detailed function reference
- [Example Workflow](/examples/mcp_workflow_example.py)
- [GitHub Repository](https://github.com/dermatologist/crisp-t)
