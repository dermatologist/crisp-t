# CRISP-T CLI Skill

## Overview

This skill enables agents to perform qualitative and mixed-methods research analysis using **CRISP-T command-line tools**. Three main CLI commands are available:

- **`crisp`** - Main analysis engine (text/NLP, ML, visualization workflows)
- **`crispt`** - Corpus management (document manipulation, semantic search, relationships)
- **`crispviz`** - Visualization generation (charts, word clouds, graphs, LDA)


## Core Commands

* **If the command is not found in your environment, try prefixing with `uv run`**
* If it still fails, ensure CRISP-T is installed: `pip install crisp-t[ml]`

## Tips
* Perform multi-step workflows STEP-BY-STEP, saving intermediate results with `--out` for analytical flexibility.
* Do not run all analyses at once; break into smaller steps to isolate issues.
* If analysis results seem off, clear cache with `--clear` before re-running.
* If a particular analysis fails or takes too long, try reducing dataset size with filters or `--num` or `--rec` or both.
* If errors persist or if it still takes too long, skip the step and proceed to the next analysis.
* Document level TOPIC assignment using `--assign` is a VERY important step different from just running `--topics`. THIS STEP MUST be for TEXT DATA.
* Use `--clear` when switching datasets or modifying filters before `--assign`
* Generate a report as you go, documenting insights from each step.
* If the source folder contains multiple CSV files, warn the user that only one CSV file is supported.
* If the folder has no text or PDF files, warn the user that at least one text or PDF file is required.

## Important steps
Collect data (e.g. in crisp_source folder). Data includes multiple text documents (txt or pdf) and ONE numeric CSV file with (optional) timestamps.
Import data into CRISP-T corpus and dataframe.
Perform linking between text and numeric data using various methods (id based, keyword based, time based, embedding based).
Explore text data using various methods (e.g., topic modeling, keyword extraction, sentiment analysis, visualizations).
Explore numeric data using various methods (e.g., summary statistics, classification, clustering, regression, association, visualizations, TDA, etc.).
Perform cross modal analysis using linked text and numeric data (e.g., text features as predictors for numeric outcomes, numeric features as predictors for text outcomes, etc.).
Add manual connections between text documents and numeric rows if needed to support theory driven analysis.
Derive insights from the analysis and document them.
---

### Reference Guide: https://r.jina.ai/https://github.com/dermatologist/crisp-t/wiki


### 1. CRISP - Main Analysis Engine

**Command**: `crisp [options]`

**Essential Workflow**:
```bash
# Step 1: Import data from a source directory
crisp --source data_folder --out corpus_output

# Step 2: Run analysis on imported corpus
crisp --inp corpus_output [analysis options]

# Step 3: Save results
crisp --inp corpus_output --out results_folder
```

#### Data Loading Options

| Option | Format | Purpose |
|--------|--------|---------|
| `--source/-s` | directory/URL | Import new data (creates corpus). Source folder should contain .txt, .pdf, .csv files |
| `--inp/-i` | path | Load existing corpus (corpus.json + corpus_df.csv) |
| `--out/-o` | path | Save corpus after analysis |
| `--unstructured/-t` | column_name | Mark CSV columns as free-text. Use multiple times for multiple columns |
| `--num/-n` | integer (default: 3) | When importing: max number of text/PDF files. When analyzing: numerical parameter (clusters, topics, etc.) |
| `--rec/-r` | integer (default: 3) | When importing: max CSV rows. When analyzing: top results to display |

**Example**:
```bash
crisp --source interview_data --unstructured responses --unstructured notes \
      --out my_corpus --num 50 --rec 10
```

#### Text Analysis Options

| Option | Flag | Purpose |
|--------|------|---------|
| `--codedict` | ✓ | Generate coding dictionary (verbs, nouns, adjectives/adverbs) |
| `--topics` | ✓ | Perform LDA topic modeling |
| `--assign` | ✓ | Assign documents to topics (run after --topics) |
| `--cat` | ✓ | Extract categories/concepts |
| `--summary` | ✓ | Generate extractive summary |
| `--sentiment` | ✓ | VADER sentiment analysis (corpus-level) |
| `--sentence` | ✓ | Document-level sentiment scores |
| `--nlp` | ✓ | Run ALL text analysis (codedict, topics, categories, summary, sentiment) |

**Important Notes**:
- Use `--clear` before `--assign` if corpus/filters changed
- VADER sentiment is corpus-level by default; combine with `--sentence` for document-level
- `--nlp` runs all text analyses sequentially

**Example**:
```bash
# Run all text analysis
crisp --inp my_corpus --nlp --out results

# Topic analysis workflow
crisp --inp my_corpus --topics --num 5 --assign --out results

# Sentiment with document-level scores
crisp --inp my_corpus --sentiment --sentence --out results
```

#### Machine Learning Options

| Option | Flag | Requires ML | Purpose |
|--------|------|------------|---------|
| `--cls` | ✓ | Yes | Classification (SVM + Decision Tree) |
| `--nnet` | ✓ | Yes | Neural Network classifier |
| `--knn` | ✓ | Yes | K-Nearest Neighbors search |
| `--kmeans` | ✓ | Yes | K-Means clustering |
| `--pca` | ✓ | Yes | Principal Component Analysis |
| `--regression` | ✓ | Yes | Linear/Logistic regression (auto-detect) |
| `--lstm` | ✓ | Yes | LSTM neural network on text |
| `--cart` | ✓ | Yes | Association rules (Apriori algorithm) |
| `--ml` | ✓ | Yes | Run ALL ML analyses |

**ML-Related Options**:
- `--outcome` (column_name or text_field) - Target variable for prediction
- `--linkage` (id/embedding/temporal/keyword) - Link text metadata to outcome
- `--aggregation` (majority/mean/first/mode) - How to combine multiple documents
- `--include` (columns) - Specific features to use
- `--ignore` (columns) - Features to exclude

**Example**:
```bash
# Classification with text metadata outcome
crisp --inp my_corpus --cls --outcome topic_name --linkage keyword --outcome outcome_column

# Regression with numeric outcome
crisp --inp my_corpus --regression --outcome satisfaction_score --include age,income

# Neural network with auto-detected outcome type
crisp --inp my_corpus --nnet --outcome survey_response --aggregation mean

# K-Means clustering with specific features
crisp --inp my_corpus --kmeans --num 4 --include age,income,years_experience
```

#### Filtering & Processing

| Option | Format | Purpose |
|--------|--------|---------|
| `--filters/-f` | key=value or link | Filter documents/rows. Multiple filters use AND logic |
| `--ignore` | comma-separated | Exclude words/columns from analysis |
| `--include` | comma-separated | Include specific columns |
| `--clear` | ✓ | Clear cache before analysis (use when switching datasets) |
| `--verbose/-v` | ✓ | Show detailed debugging information |

**Filter Examples**:
```bash
# Metadata filters
crisp --inp corpus --filters "region=North" --filters "source=Interview" --nlp

# Link-based filters (requires prior linking)
crisp --inp corpus --filters "embedding:text" --filters "temporal:df" --sentiment

# Combined
crisp --inp corpus --filters "sentiment=positive" --topics --num 5
```

#### Output & Display

| Option | Format | Purpose |
|--------|--------|---------|
| `--print/-p` | documents/N | Display corpus info. Examples: --print documents, --print 10 |
| `--sources` | path | Load from multiple source folders (used multiple times) |

---

### 2. CRISPT - Corpus Management

**Command**: `crispt [options]`

**Purpose**: Manipulate corpus structure, documents, metadata, and analyze semantic relationships.

#### Corpus Creation & Management

| Option | Format | Purpose |
|--------|--------|---------|
| `--id` | text (required for new) | Unique corpus identifier |
| `--name` | text | Descriptive corpus name |
| `--description` | text | Detailed corpus description |
| `--inp` | path | Load existing corpus |
| `--out` | path | Save corpus |
| `--print` | ✓ | Display full corpus |
| `--clear-rel` | ✓ | Remove all relationships |
| `--verbose/-v` | ✓ | Debug mode |

**Example**:
```bash
# Create new corpus
crispt --id my_study --name "Health Interview Study" --description "2025 interviews" \
       --out corpus_folder

# Load and display
crispt --inp corpus_folder --print

# Clear relationships
crispt --inp corpus_folder --clear-rel --out corpus_folder
```

#### Document Operations

| Option | Format | Purpose |
|--------|--------|---------|
| `--doc` | id\|name\|text | Add document (name optional) |
| `--remove-doc` | doc_id | Remove document by ID |
| `--doc-ids` | ✓ | List all document IDs |
| `--doc-id` | doc_id | Display specific document details |
| `--meta` | key=value | Add corpus metadata |

**Document Format**: `id|name|text` or `id|text`

**Example**:
```bash
# Add documents
crispt --id study --doc "interview1|Interview with Jane|Interview transcript..." \
       --doc "interview2|Interview with Bob|..." --out corpus_folder

# View documents
crispt --inp corpus_folder --doc-ids
crispt --inp corpus_folder --doc-id interview1

# Remove document
crispt --inp corpus_folder --remove-doc interview2 --out corpus_folder
```

#### Relationship Management

| Option | Format | Purpose |
|--------|--------|---------|
| `--add-rel` | first\|second\|relation | Add text↔numeric relationship |
| `--print-relationships` | ✓ | Show all relationships |
| `--relationships-for-keyword` | keyword | Find relationships involving keyword |

**Relationship Format**: `first|second|relation`
- **first**: `text:keyword` or `num:column`
- **second**: `text:keyword` or `num:column`
- **relation**: `correlates`, `predicts`, `contrasts`, etc.

**Example**:
```bash
# Add relationships after topic modeling
crispt --inp corpus_folder \
       --add-rel "text:healthcare|num:satisfaction_score|predicts" \
       --add-rel "text:cost_barriers|num:income_level|correlates" \
       --out corpus_folder

# Display relationships
crispt --inp corpus_folder --print-relationships
crispt --inp corpus_folder --relationships-for-keyword healthcare
```

#### Semantic Search Operations

| Option | Format | Purpose |
|--------|--------|---------|
| `--semantic` | query_text | Find documents similar to query |
| `--similar-docs` | doc_id1,doc_id2 | Find docs similar to reference docs |
| `--semantic-chunks` | query_text | Search within document chunks |
| `--doc-id` | doc_id | Specify document for chunk search |
| `--num` | integer (default: 5) | Results to return |
| `--rec` | float (default: 0.4) | Similarity threshold 0-1 |
| `--metadata-df` | ✓ | Export search metadata to DataFrame |
| `--metadata-keys` | keys | Specific metadata to export |

**Example**:
```bash
# Semantic search
crispt --inp corpus_folder --semantic "healthcare barriers" --num 10

# Find similar documents (literature review snowballing)
crispt --inp corpus_folder --similar-docs "doc1,doc2" --num 20 --rec 0.7

# Search within document
crispt --inp corpus_folder --semantic-chunks "cost barriers" \
       --doc-id interview1 --rec 0.5

# Export metadata
crispt --inp corpus_folder --metadata-df --metadata-keys "source,date,region"
```

#### DataFrame Operations

| Option | Format | Purpose |
|--------|--------|---------|
| `--df-cols` | ✓ | Show all DataFrame column names |
| `--df-row-count` | ✓ | Show row count |
| `--df-row` | index | Display specific row |

**Example**:
```bash
crispt --inp corpus_folder --df-cols
crispt --inp corpus_folder --df-row-count
crispt --inp corpus_folder --df-row 0
```

#### Temporal Analysis

| Option | Format | Purpose |
|--------|--------|---------|
| `--temporal-link` | method:column[:param] | Link documents to rows by time |
| `--temporal-filter` | start:end | Filter by time range (ISO 8601) |
| `--temporal-summary` | period | Summarize by time period |

**Methods**:
- `nearest:column` - Nearest timestamp
- `window:column:seconds` - Within time window
- `sequence:column:period` - By periods (D/W/M/Y)

**Example**:
```bash
# Link by nearest time
crispt --inp corpus_folder --temporal-link "nearest:timestamp" --out corpus_folder

# Link with 5-minute window
crispt --inp corpus_folder --temporal-link "window:timestamp:300" --out corpus_folder

# Link weekly
crispt --inp corpus_folder --temporal-link "sequence:timestamp:W" --out corpus_folder

# Filter time range
crispt --inp corpus_folder --temporal-filter "2025-01-01:2025-06-30" --out filtered_corpus

# Weekly summary
crispt --inp corpus_folder --temporal-summary "W"
```

#### Advanced Analysis

| Option | Format | Purpose |
|--------|--------|---------|
| `--tdabm` | y:x_vars[:radius] | Topological Data Analysis Ball Mapper |
| `--graph` | ✓ | Generate corpus relationship graph |

**Example**:
```bash
# TDABM analysis
crispt --inp corpus_folder --tdabm "satisfaction:age,income:0.3" --out corpus_folder

# Generate graph
crispt --inp corpus_folder --graph --out corpus_folder
```

---

### 3. CRISPVIZ - Visualization Engine

**Command**: `crispviz [options]`

**Purpose**: Generate charts, word clouds, LDA visualizations, and relationship graphs.

#### Basic Options

| Option | Format | Purpose |
|--------|--------|---------|
| `--inp/-i` | path | Load corpus for visualization |
| `--out/-o` | path | Output directory for images |
| `--bins` | integer (default: 100) | Bins for frequency histograms |
| `--topics-num` | integer (default: 8) | Number of LDA topics |
| `--top-n` | integer (default: 20) | Top terms to display |
| `--verbose/-v` | ✓ | Debug output |

#### Visualization Types

| Option | Flag | Requires | Purpose |
|--------|------|----------|---------|
| `--freq` | ✓ | None | Word frequency distribution |
| `--top-terms` | ✓ | None | Top terms bar chart |
| `--wordcloud` | ✓ | LDA topics | Topic word cloud |
| `--ldavis` | ✓ | LDA topics | Interactive LDA visualization (HTML) |
| `--by-topic` | ✓ | LDA topics | Distribution by dominant topic |
| `--corr-heatmap` | ✓ | Numeric data | Correlation matrix heatmap |
| `--tdabm` | ✓ | TDABM analysis | TDABM topology visualization |
| `--graph` | ✓ | Graph data | Relationship network graph |

#### Graph Visualization Options

| Option | Format | Purpose |
|--------|--------|---------|
| `--graph-nodes` | node_types | Node types: document, keyword, cluster, metadata |
| `--graph-layout` | algorithm | Layout: spring (default), circular, kamada_kawai, spectral |

#### Correlation Heatmap Options

| Option | Format | Purpose |
|--------|--------|---------|
| `--corr-columns` | column_list | Specific numeric columns (auto-selected if empty) |

**Example Usage**:
```bash
# Word frequency
crispviz --inp corpus_folder --out viz_output --freq

# Top terms chart
crispviz --inp corpus_folder --out viz_output --top-terms --top-n 30

# LDA visualizations (requires prior topic modeling)
crispviz --inp corpus_folder --out viz_output --wordcloud
crispviz --inp corpus_folder --out viz_output --ldavis --topics-num 5
crispviz --inp corpus_folder --out viz_output --by-topic

# Correlation analysis
crispviz --inp corpus_folder --out viz_output --corr-heatmap \
         --corr-columns age,income,satisfaction_score

# Network graph
crispviz --inp corpus_folder --out viz_output --graph \
         --graph-nodes document,keyword --graph-layout spring

# TDABM topology
crispviz --inp corpus_folder --out viz_output --tdabm

# All visualizations
crispviz --inp corpus_folder --out viz_output --freq --top-terms \
         --wordcloud --corr-heatmap --graph
```

---

## Common Workflows

### Workflow 1: Basic Qualitative Analysis

```bash
# 1. Import data
crisp --source research_data --out corpus --num 100 --unstructured "open_ended_q"

# 2. Generate coding dictionary
crisp --inp corpus --codedict --out corpus_v1

# 3. Topic modeling
crisp --inp corpus_v1 --topics --num 5 --assign --out corpus_v2

# 4. Sentiment analysis
crisp --inp corpus_v2 --sentiment --sentence --out corpus_v3

# 5. Visualizations
crispviz --inp corpus_v3 --out visualizations --freq --wordcloud --by-topic

# 6. Save final corpus
crispt --inp corpus_v3 --out final_corpus --print
```

### Workflow 2: Mixed-Methods Triangulation

```bash
# 1. Create corpus with CSV data
crisp --source data --unstructured comments --out corpus

# 2. Generate text analysis
crisp --inp corpus --topics --num 4 --assign --sentiment --out corpus_analyzed

# 3. Add relationships linking text findings to numeric outcomes
crispt --inp corpus_analyzed \
       --add-rel "text:healthcare|num:satisfaction_score|predicts" \
       --add-rel "text:cost_concerns|num:household_income|correlates" \
       --out corpus_linked

# 4. ML analysis linking text to numeric
crisp --inp corpus_linked --regression --outcome satisfaction_score \
      --linkage keyword --aggregation mean --out results

# 5. Visualize relationships
crispviz --inp results --out viz --graph --corr-heatmap
crispt --inp results --print-relationships
```

### Workflow 3: Temporal Analysis

```bash
# 1. Import time-stamped data
crisp --source time_series_data --out corpus

# 2. Link documents by time
crispt --inp corpus --temporal-link "sequence:timestamp:W" --out corpus_temporal

# 3. Generate temporal summary
crispt --inp corpus_temporal --temporal-summary "W"

# 4. Filter to specific period
crispt --inp corpus_temporal --temporal-filter "2025-01-01:2025-06-30" --out corpus_period

# 5. Analyze sentiment over time
crisp --inp corpus_period --sentiment --sentence --out results_period

# 6. Visualize time series
crispviz --inp results_period --out viz --freq --by-topic
```

### Workflow 4: ML Classification with Mixed Data

```bash
# 1. Prepare corpus
crisp --source data --unstructured text_col --out corpus

# 2. Generate text features
crisp --inp corpus --codedict --topics --num 3 --assign --out corpus_features

# 3. Train classifier with text metadata
crisp --inp corpus_features --cls --outcome satisfaction_level \
      --linkage keyword --aggregation majority --include age,income \
      --out classifier_results

# 4. View feature importance
crispt --inp classifier_results --print-relationships

# 5. Visualize results
crispviz --inp classifier_results --out viz --corr-heatmap --graph
```

---

## Key Concepts for Agents

### Corpus Structure
- **Documents**: Text entries (interviews, field notes, etc.)
- **DataFrame**: Numeric data (age, income, survey responses, etc.)
- **Relationships**: Explicit links between text findings and numeric variables
- **Metadata**: Tags, timestamps, source information

### Linkage Methods
- **id**: Direct document-to-row matching by ID
- **embedding**: Semantic similarity-based linking
- **temporal**: Time-based linking (nearest, window, sequence)
- **keyword**: Linking via extracted keywords/topics

### Aggregation Strategies
- **majority**: Most common value (classification)
- **mean**: Average value (regression)
- **first**: First value encountered
- **mode**: Most frequent value

### Important Flags
- `--clear`: Always use before `--assign` if filters/data changed
- `--linkage`: Required when outcome is a text field
- `--unstructured`: Mark free-text columns in CSV for proper analysis
- `--verbose`: Essential for debugging multi-step workflows

### File Formats
- **Corpus files**: `corpus.json` + `corpus_df.csv` (created in `--out` folder)
- **Visualizations**: PNG/HTML (saved to `--out` folder)
- **Metadata**: Embedded in corpus.json (view with `--print`)

---

## Tips for Effective Use

1. **Always start with `--source`** to import data into a corpus structure
2. **Use `--clear`** when switching datasets or modifying filters
3. **Combine `--nlp`** to run all text analyses at once
4. **Save intermediate results** using `--out` at each major step
5. **Use filtering** (`--filters`) to analyze subsets before full analysis
6. **Link early**: Add relationships after text analysis for mixed-methods validation
7. **Visualize often**: Use `crispviz` after each major analysis step
8. **Check metadata**: Use `crispt --print` to inspect corpus structure

---

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| `Cache error before --assign` | Cache from previous run | Use `--clear` flag |
| `Outcome not found` | Wrong column/field name | Use `crispt --df-cols` or `crispt --print` to verify |
| `ML features mismatch` | Features changed after training | Clear cache and retrain |
| `Linkage failed` | Insufficient data/metadata | Verify timestamps or use simpler linkage method |
| `Visualization empty` | Analysis not run | Ensure `--topics`, `--tdabm`, or `--graph` completed first |

---

## Performance Notes

- **Large corpora** (1000+ docs): Use `--num` to limit imports, use filters
- **Topic modeling**: Adjust `--num` lower for faster processing (3-5 recommended)
- **TDABM/graphs**: More expensive; save intermediate results
- **Semantic search**: Requires initialization; slower on first run
- **ML training**: Very slow on large datasets; use sampling/filtering

