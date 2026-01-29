# CRISP-T CLI Cheatsheet

This cheatsheet provides a comprehensive reference for all Command Line Interface (CLI) commands available in CRISP-T.

## 📋 Command Overview

CRISP-T provides three main CLI tools:
1. **`crisp`**: Main tool for data import, analysis (text & numeric), and machine learning.
2. **`crispt`**: Corpus management tool for detailed structural manipulation, metadata editing, and semantic search.
3. **`crispviz`**: Visualization tool for generating charts, graphs, and word clouds.

---

## 📥 Data Import Commands (Required first step)

Used to bring data into the CRISP-T environment.

| Command | Description | Example |
| :--- | :--- | :--- |
| `crisp --source <path>` | Import data from a directory (PDFs, TXT, CSV) or URL. | `crisp --source ./raw_data --out ./crisp_input` |
| `crisp --sources <path> --sources <path2>` | Import from multiple sources. | `crisp --sources ./data1 --sources ./data2` |
| `crisp --unstructured <col>` | Specify CSV columns containing free text to treat as documents. | `crisp --source ./data --unstructured "comments"` |

**Common Options:**
*   `--out <path>`: Directory to save the imported corpus (required). Recommended value `crisp_input`.
*   `--num <int>`: Limit number of text files to import (when used with `--source`).
*   `--rec <int>`: Limit number of CSV rows to import (when used with `--source`).
*   `--ignore <text>`: Comma-separated list of words/columns to exclude during import.

---

## 🔍 Data Analysis Commands

**All commands below are run with the `--inp <corpus-path>` argument to specify the input corpus. `--inp` defaults to `crisp_input`.**

### Text Analysis (NLP)

| Action | Flag | Description |
| :--- | :--- | :--- |
| **Topic Modeling** | `--topics` | Run Latent Dirichlet Allocation (LDA) to find topics. |
| **Assign Topics** | `--assign` | Assign the most relevant topic to each document. |
| **Sentiment** | `--sentiment` | VADER sentiment analysis (pos, neg, neu, compound). |
| **Summarize** | `--summary` | Generate an extractive summary of the corpus. |
| **Categories** | `--cat` | Extract common categories/themes. |
| **Coding Dictionary**| `--codedict` | Generate a qualitative coding dictionary. |
| **All NLP** | `--nlp` | Run *all* above NLP analyses at once. |

*   `--visualize`: Prepare data for visualization; used with `--assign` (topic assignments) and with `crispviz`.

### Numeric & Statistical Analysis

| Action | Flag | Description |
| :--- | :--- | :--- |
| **Clustering** | `--kmeans` | K-Means clustering of numeric data. |
| **Regression** | `--regression` | Linear or Logistic regression (auto-detected). |
| **Classification** | `--cls` | Decision Tree and SVM classification. |
| **Neural Net** | `--nnet` | Simple Neural Network classifier. |
| **LSTM** | `--lstm` | Long Short-Term Memory network for text sequences. |
| **Association Rules**| `--cart` | Apriori algorithm for association rules. |
| **PCA** | `--pca` | Principal Component Analysis dimensionality reduction. |
| **Nearest Neighbors**| `--knn` | K-Nearest Neighbors search. |
| **All ML** | `--ml` | Run *all* above ML analyses (requires `crisp-t[ml]`). |

**Common Analysis Options:**
*   `--num <n>`: Parameter for analysis (e.g., number of clusters, topics, or summary sentences). Default: 3.
*   `--rec <n>`: Number of results/rows to display. Default: 3.
*   `--filters <key=val>`: Filter data before analysis (e.g., `category=A`).
*   `--outcome <col>`: Target variable for ML/Regression.
*   `--include <cols>`: specific columns to include in analysis.
*   `--aggregation <strategy>`: Aggregation strategy when multiple documents link to one row. Choices: `majority`, `mean`, `first`, `mode`. (Used with cross-modal analysis. [More details](../notes/text_metadata_outcomes_ml.md)).

---

## 📊 Visualization Commands (`crispviz`)

Run these commands *after* performing the relevant analysis with `crisp` or `crispt`.

| Visualization | Flag | Prerequisite |
| :--- | :--- | :--- |
| **Word Frequency** | `--freq` | None |
| **Top Terms** | `--top-terms` | None |
| **Correlation Heatmap**| `--corr-heatmap`| CSV data present |
| **Topic Word Cloud** | `--wordcloud` | `crisp --topics` |
| **LDA Interactive** | `--ldavis` | `crisp --topics` (Saves as HTML) |
| **Topic Distribution** | `--by-topic` | `crisp --topics` |
| **TDABM Network** | `--tdabm` | `crispt --tdabm ...` |
| **Knowledge Graph** | `--graph` | `crispt --graph` |

**Common Options:**
*   `--out <dir>`: **Required**. Directory to save images (PNG/HTML).
*   `--inp <dir>`: Input corpus directory.
*   `--topics-num <n>`: Number of topics to assume for visualization (default: 8).
*   `--bins <n>`: Number of bins for histograms (default: 100).
*   `--top-n <n>`: Number of top terms to display in `--top-terms` (default: 20).
*   `--corr-columns <col1,col2>`: Comma-separated numeric columns to use for `--corr-heatmap`; otherwise columns are auto-selected.
*   `--graph-nodes <types>`: Comma-separated node types to include in graph viz: `document`, `keyword`, `cluster`, `metadata` (use `all` for every type).
*   `--graph-layout <name>`: Layout algorithm for graph viz: `spring` (default), `circular`, `kamada_kawai`, or `spectral`.

---

## 🛠 Corpus Manipulation (`crispt`)

Advanced tools for managing the corpus structure.

### Search & Query
*   `--semantic "query"`: Semantic search for documents.
*   `--semantic-chunks "query"`: Search within specific document chunks (needs `--doc-id`).
*   `--similar-docs "id1,id2"`: Find documents similar to the provided IDs.
*   `--feature-search "query"`: Search for features/variables in the dataframe.

**Inspect & Utilities:**
*   `--print`: Display the full corpus structure in a formatted view.
*   `--df-cols`: List all DataFrame column names.
*   `--df-row-count`: Show the number of rows in the DataFrame.
*   `--df-row <index>`: Display a specific DataFrame row by index.
*   `--doc-ids`: List all document IDs in the corpus.
*   `--doc-id <id>`: Show details for a specific document by ID.
*   `--remove-doc <id>`: Remove a document from the corpus by ID.

**Print Options:**
*   Two formats are supported for `--print`:
	- Multiple flags: pass several `--print` flags, e.g. `--print documents --print 10`.
	- Single string: pass the full command as one quoted argument, e.g. `--print "documents 10"` or `--print "dataframe metadata"`.

Examples:
```bash
# Print first 10 documents (two-flag form)
crispt --inp ./corpus --print documents --print 10

# Print DataFrame metadata columns (single-string form)
crispt --inp ./corpus --print "dataframe metadata"
```

### Metadata & Structure
*   `--id <name>`: Create new corpus with ID.
*   `--doc "id|text"`: Add a document manually.
*   `--meta "key=val"`: Add metadata to corpus.
*   `--add-rel "A|B|rel"`: Define relationship between data points.
*   `--temporal-link <method>`: Link docs to data by time (`nearest`, `window`, etc.).
*   `--clear-rel`: Remove all relationships from the corpus metadata.
*   `--relationships`: Print all relationships defined in the corpus.
*   `--relationships-for-keyword <kw>`: Print relationships involving a specific keyword.
*   `--metadata-df`: Export semantic-search collection metadata as a DataFrame.
*   `--metadata-keys <keys>`: Comma-separated metadata keys to include when exporting.

### Advanced Analysis
*   `--tdabm "y:x1,x2:ranking"`: Topological Data Analysis / Agent Based Modeling.
*   `--graph`: Generate graph network from keywords/metadata.

**Temporal & Embedding Tools:**
*   `--temporal-summary <period>`: Generate temporal summary (`D`, `W`, `M`, `Y`).
*   `--temporal-sentiment <period:aggregation>`: Analyze sentiment trends (e.g., `W:mean`).
*   `--temporal-topics <period:top_n>`: Extract topics over time (e.g., `W:5`).
*   `--temporal-subgraphs <period>`: Create time-sliced subgraphs (e.g., `W`).
*   `--embedding-link <metric:top_k:threshold>`: Link by embedding similarity (e.g., `cosine:1:0.7`).
*   `--embedding-stats`: Display statistics about embedding-based links.
*   `--embedding-viz <method:output_path>`: Visualize embedding space (`tsne`, `pca`, `umap`).

---

## 🔗 Data Linking & Filtering

**Linking Text to Numbers:**
*   **Keyword Linking**: Default. Matches text keywords to dataframe columns.
*   **ID Linking**: Use `--linkage id` to match Document ID to a DataFrame column.
*   **Temporal Linking**: `crispt --temporal-link "window:timestamp:300"` links by time proximity (in seconds). [More details](../notes/TEMPORAL_ANALYSIS.md)
*   **Embedding Linking**: `crispt --embedding-link "cosine:1:0.7"` [More details](../notes/EMBEDDING_LINKING.md)

**Filtering:**
*   `--filters key=value` or `--filters key:value`: Standard exact match filter. Both `=` and `:` are accepted as separators for key/value filters.
*   Special link filters: `--filters embedding:text`, `--filters embedding:df`, `--filters temporal:text`, `--filters temporal:df` (you can also use `=` instead of `:`). These select by linked rows or documents using embedding or temporal links.
*   Legacy shorthand mappings: `--filters =embedding` or `--filters :embedding` are mapped to `embedding:text`; `--filters =temporal` or `--filters :temporal` are mapped to `temporal:text`.
*   ID linkage: `--filters id=<value>` filters to a specific document/row by ID. Using `--filters id:` or `--filters id=` with no value syncs remaining documents and dataframe rows by matching `id` values after other filters are applied.
*   `--temporal-filter "start:end"`: Filter by time range (ISO format).

---

## 💡 Common Workflows

### 1. Basic Qualitative Analysis
```bash
# Import
crisp --source ./interviews --out ./corpus
# Analyze Topics & Sentiment
crisp --inp ./corpus --topics --sentiment --out ./corpus_analyzed
# Visualize
crispviz --inp ./corpus_analyzed --wordcloud --ldavis --out ./viz
```

### 2. Mixed Methods (Text + CSV)
```bash
# Import CSV with text column
crisp --source ./survey_data --unstructured "comments" --out ./survey_corpus
# Cluster numeric data & analyze text options
crisp --inp ./survey_corpus --kmeans --num 5 --topics --num 5
```

### 3. Semantic Search
```bash
# Find documents about specific topic
crispt --inp ./corpus --semantic "patient anxiety" --num 10
```

### 4. Cross-Modal Analysis (Text + CSV)
```bash
# Import CSV that contains free-text comments and numeric outcomes
crisp --source ./survey_data --unstructured "comments" --out ./survey_corpus

# Link documents to rows by ID, then run classification using linked text as features
crisp --inp ./survey_corpus --linkage id --outcome satisfaction_score --cls --aggregation majority

# Use embedding linking to aggregate document embeddings to rows then run regression
crispt --inp ./survey_corpus --filters embedding:text --out ./linked && \
crisp --inp ./linked --outcome satisfaction_score --regression
```

---

## ⚠️ Troubleshooting

*   **"No input data provided"**: Ensure you used `--source` (to import) or `--inp` (to load existing).
*   **ML dependencies missing**: Run `pip install crisp-t[ml]`.
*   **Visualization errors**: Ensure you ran the prerequisite analysis step first (e.g., running `--topics` before `--wordcloud`).
*   **Caching issues**: Use `--clear` to force a refresh if you change datasets or experience weird results.
