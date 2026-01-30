# CRISP-T CLI Agent Skill - README

## What is This?

This skill provides AI agents with comprehensive knowledge of CRISP-T's command-line interface (CLI) tools for qualitative and mixed-methods research analysis. It enables agents to orchestrate complex analysis workflows combining text analysis, machine learning, semantic search, and data visualization.

## Three Main Tools

### 1. **crisp** - Analysis Engine
The main tool for data import, text/NLP analysis, and machine learning.

```bash
crisp --source data --out corpus --nlp --ml --visualize
```

**Primary Uses**:
- Import text (.txt, .pdf) and numeric data (.csv)
- Generate coding dictionaries, topics, sentiment analysis
- Train classifiers, regressions, clustering models
- Link text findings to numeric outcomes

### 2. **crispt** - Corpus Manager
Manipulate corpus structure, documents, semantic relationships, and metadata.

```bash
crispt --inp corpus --add-rel "text:theme|num:score|predicts" --out corpus
```

**Primary Uses**:
- Manage documents (add, remove, display)
- Create explicit relationships between text and numeric data
- Perform semantic search (find similar documents)
- Temporal analysis (time-based linking)
- TDABM analysis (topological data analysis)

### 3. **crispviz** - Visualization Engine
Generate publication-ready visualizations from analyzed corpora.

```bash
crispviz --inp corpus --out viz --wordcloud --graph --corr-heatmap
```

**Primary Uses**:
- Word frequency distributions
- Topic visualizations (word clouds, LDA)
- Network graphs (document-keyword relationships)
- Correlation heatmaps
- TDABM topology maps

## Quick Start for Agents

### Step 1: Import Data
```bash
crisp --source my_data_folder \
      --unstructured responses \
      --out my_corpus \
      --num 100 --rec 10
```
Creates a corpus from .txt, .pdf, and .csv files in `my_data_folder`.

### Step 2: Analyze
```bash
crisp --inp my_corpus --nlp --sentiment --topics --num 5 --out my_corpus_v2
```
Runs comprehensive text analysis on the corpus.

### Step 3: Link & Relate
```bash
crispt --inp my_corpus_v2 \
       --add-rel "text:healthcare|num:satisfaction|predicts" \
       --out my_corpus_v3
```
Explicitly link text findings to numeric outcomes.

### Step 4: ML & Triangulation
```bash
crisp --inp my_corpus_v3 --regression --outcome satisfaction_score \
      --linkage keyword --aggregation mean --out results
```
Use ML to validate text↔numeric relationships.

### Step 5: Visualize
```bash
crispviz --inp results --out visualizations --graph --wordcloud --corr-heatmap
```
Generate publication-ready visualizations.

## Key Features

### Text Analysis Capabilities
- **Coding Dictionary**: Extract verbs (actions), nouns (concepts), adjectives (qualities)
- **Topic Modeling**: LDA-based topic discovery with document assignment
- **Sentiment Analysis**: VADER sentiment at corpus or document level
- **Category Extraction**: Bag-of-terms concept identification
- **Summarization**: Extractive summaries of corpus

### Machine Learning Options
- **Classification**: SVM, Decision Trees, Neural Networks
- **Regression**: Linear/Logistic (auto-detected)
- **Clustering**: K-Means, KNN search
- **Dimensionality Reduction**: PCA
- **Pattern Mining**: Apriori algorithm (association rules)
- **Deep Learning**: LSTM for text prediction

### Semantic Analysis
- **Semantic Search**: Find documents by meaning (not keywords)
- **Similar Document Discovery**: Literature review snowballing
- **Chunk-level Search**: Find passages within documents
- **Metadata Export**: Extract search results for analysis

### Temporal Analysis
- **Time-based Linking**: Connect text to events/measurements by time
- **Temporal Filtering**: Subset by date ranges
- **Temporal Summaries**: Aggregate statistics over periods
- **Methods**: Nearest time, time windows, period-based grouping

### Advanced Analysis
- **TDABM**: Topological Data Analysis Ball Mapper (uncover hidden patterns)
- **Graph Generation**: Network visualizations of document-keyword relationships
- **Filtering**: Metadata and link-based corpus subsetting

## Use Cases

### Academic Research
- Analyze interview transcripts with coding dictionary
- Validate themes against survey responses (triangulation)
- Generate publication-ready visualizations
- Track sentiment evolution in longitudinal studies

### Market Research
- Cluster customer feedback by themes
- Link sentiment to demographic characteristics
- Predict satisfaction from coded responses
- Generate executive summaries

### Clinical Research
- Extract coding dictionaries from patient narratives
- Correlate qualitative codes with clinical outcomes
- Temporal analysis of symptom reports
- Network analysis of comorbidities

### Policy Analysis
- Topic modeling of policy documents
- Relationship analysis between concepts and metrics
- Temporal evolution of policy themes
- Stakeholder network visualization

## Workflow Examples

### Mixed-Methods Example
```bash
# 1. Import
crisp --source research --out corpus

# 2. Text analysis
crisp --inp corpus --topics --num 4 --assign --sentiment --out corpus_v1

# 3. Create relationships
crispt --inp corpus_v1 \
       --add-rel "text:barriers|num:completion_rate|predicts" \
       --add-rel "text:cost_concerns|num:family_income|correlates" \
       --out corpus_v2

# 4. ML validation
crisp --inp corpus_v2 --regression --outcome completion_rate \
      --linkage keyword --out results

# 5. Visualize
crispviz --inp results --out viz --graph --corr-heatmap --wordcloud
```

### Temporal Analysis Example
```bash
# 1. Import time-series data
crisp --source time_data --out corpus

# 2. Link by time
crispt --inp corpus --temporal-link "sequence:date:W" --out corpus_time

# 3. Filter period
crispt --inp corpus_time --temporal-filter "2025-01-01:2025-06-30" --out period

# 4. Analyze
crisp --inp period --sentiment --topics --num 3 --out period_results

# 5. Visualize trends
crispviz --inp period_results --out viz --freq --by-topic --wordcloud
```

## Agent Decision Points

Agents should consider:

1. **Data Type**: Does the corpus have both text and numeric data? Use relationships + ML.
2. **Goal**: Exploration (codedict, topics) vs Prediction (ML) vs Understanding (graphs)?
3. **Complexity**: Start simple (codedict) → build complexity (topics, ML, linking)
4. **Filtering**: Should analysis focus on a subset? Use `--filters` or `--temporal-filter`
5. **Validation**: Need triangulation? Use `--linkage` and `--aggregation` to connect text↔numeric

## Common Parameters Across Tools

| Parameter | Purpose | Values | Example |
|-----------|---------|--------|---------|
| `--inp` | Load corpus | folder path | `--inp my_corpus` |
| `--out` | Save output | folder path | `--out results` |
| `--num` | Count/limit | integer | `--num 5` |
| `--rec` | Results/threshold | integer or 0-1 | `--rec 0.7` |
| `--filters` | Subset data | key=value or link | `--filters "region=North"` |
| `--outcome` | Target variable | column/field name | `--outcome satisfaction` |
| `--linkage` | Text-to-outcome method | id/embedding/temporal/keyword | `--linkage keyword` |
| `--clear` | Clear cache | flag | `--clear` |
| `--verbose` | Debug output | flag | `--verbose` |

## Important Concepts

### Corpus Files
- **corpus.json**: Document text + metadata + analysis results
- **corpus_df.csv**: Numeric data (if CSV provided)
- Both created when using `--out`

### Relationships Format
`first|second|relation` where:
- first/second: `text:keyword` or `num:column`
- relation: `predicts`, `correlates`, `contrasts`, etc.

### Analysis Modes
- **Text-centric**: Start with `--codedict`, `--topics`, `--sentiment`
- **ML-centric**: Requires `--outcome` and features (`--include`)
- **Mixed-methods**: Combines both via `--linkage` and `--aggregation`

## Performance Considerations

- **Large datasets**: Use `--num` to limit imports during development
- **Topics**: 3-5 topics for quick iterations, 5-10 for thorough analysis
- **Semantic search**: Slower on first run (initializes embeddings)
- **TDABM/Graphs**: Most computationally expensive, save intermediate results
- **ML training**: Very slow on 10K+ rows; use filtering/sampling

## When to Use Each Tool

| Task | Tool | Command |
|------|------|---------|
| Import data | crisp | `--source ... --out` |
| Text analysis | crisp | `--codedict`, `--topics`, `--sentiment` |
| ML training | crisp | `--cls`, `--regression`, `--kmeans` |
| Add documents | crispt | `--doc` |
| Create relationships | crispt | `--add-rel` |
| Semantic search | crispt | `--semantic` |
| Temporal analysis | crispt | `--temporal-link`, `--temporal-filter` |
| Visualization | crispviz | `--freq`, `--wordcloud`, `--graph` |

## Troubleshooting

**Problem**: `Cache error before --assign`
- **Solution**: Add `--clear` flag: `crisp --clear --inp corpus --assign`

**Problem**: `Outcome not found`
- **Solution**: Check column name: `crispt --inp corpus --df-cols` or `--print`

**Problem**: `Linkage failed`
- **Solution**: Verify timestamps or ensure analysis was run: `crispt --inp corpus --print`

**Problem**: `Slow performance`
- **Solution**: Use `--num` to limit data, add `--filters` to subset, or filter time range

## Resources

- **Full documentation**: See SKILL.md for detailed parameter references
- **Example workflows**: See examples/ folder in repository
- **Source code**: crisp.py, corpuscli.py, vizcli.py in src/crisp_t/

