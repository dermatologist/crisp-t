# Using Text Metadata Fields as ML Outcome Variables

This guide demonstrates how to use text metadata fields as outcome variables in machine learning tasks when text documents are linked to numeric data through various linkage methods.

## Overview

CRISP-T now supports using text metadata fields (qualitative codes from document analysis) as outcome variables for machine learning predictions. This enables:

1. **Triangulation**: Link qualitative findings (document metadata) with quantitative analysis
2. **Validation**: Test if numeric variables can predict qualitative codes
3. **Integration**: Seamlessly combine text analysis results with ML workflows

## Linkage Methods

Four linkage methods connect text documents to numeric DataFrame rows:

### 1. ID Linkage (`id`)
- **Mechanism**: Direct matching via document.id and df['id'] column
- **Use Case**: When documents and rows share explicit identifiers
- **Example**: Survey responses where text IDs match numeric row IDs

### 2. Embedding Linkage (`embedding`)
- **Mechanism**: Semantic similarity via sentence embeddings
- **Use Case**: When no explicit IDs exist but semantic relationships can be inferred
- **Example**: Linking patient notes to sensor data by semantic content
- **Prerequisite**: Run `embedding_linker` to create links

### 3. Temporal Linkage (`temporal`)
- **Mechanism**: Time-based proximity matching via timestamps
- **Use Case**: Time-series data where documents and rows are related by time
- **Example**: Linking diary entries to health metrics by timestamp
- **Prerequisite**: Run `temporal` linker to create links

### 4. Keyword Linkage (`keyword`)
- **Mechanism**: Keyword/tag-based matching
- **Use Case**: When documents are tagged with keywords matching DataFrame columns
- **Example**: Linking articles tagged with topics to demographic data
- **Prerequisite**: Create keyword_links in document metadata

## Aggregation Strategies

When multiple documents link to the same DataFrame row, aggregation determines the outcome value:

| Strategy | Description | Best For |
|----------|-------------|----------|
| `majority` | Most common value (majority vote) | Classification tasks (default) |
| `mean` | Average of numeric values | Regression tasks |
| `first` | First document's value | When order matters |
| `mode` | Most frequent value | Categorical outcomes |

## Command-Line Interface (CLI)

### Basic Usage Pattern

```bash
crisp --source <data_folder> \
      --outcome <text_metadata_field> \
      --linkage <linkage_method> \
      --aggregation <aggregation_strategy> \
      --<ml_task>
```

### Example 1: Regression with ID Linkage

Predict sentiment scores from text metadata using numeric features:

```bash
# Assume documents have metadata field 'sentiment_category' (positive/negative)
# and df has 'id' column matching document IDs

crisp --source ./patient_data \
      --outcome sentiment_category \
      --linkage id \
      --aggregation majority \
      --regression \
      --include "age,bmi,blood_pressure,exercise_hours"
```

**What happens:**
1. Extracts `sentiment_category` from document metadata
2. Matches documents to DataFrame rows via ID linkage
3. Uses majority vote for rows with multiple documents
4. Runs logistic regression (binary outcome detected)
5. Shows which numeric features predict sentiment

### Example 2: Neural Network with Embedding Linkage

Classify topic labels from text analysis using numeric variables:

```bash
# First, create embedding links
crisp --source ./research_data \
      --embedding-link \
      --embedding-method sentence-transformers \
      --top-k 1 \
      --out ./linked_corpus

# Then use text metadata as outcome
crisp --inp ./linked_corpus \
      --outcome primary_topic \
      --linkage embedding \
      --aggregation majority \
      --nnet \
      --include "funding_amount,team_size,citations,impact_score"
```

### Example 3: Decision Tree with Temporal Linkage

Predict emotion codes from diary entries using health metrics:

```bash
# First, create temporal links
crisp --source ./health_diary \
      --temporal-link \
      --method nearest \
      --time-column timestamp \
      --out ./linked_corpus

# Then predict emotion from health metrics
crisp --inp ./linked_corpus \
      --outcome emotion_code \
      --linkage temporal \
      --aggregation mode \
      --cls \
      --include "heart_rate,sleep_hours,step_count,stress_level"
```

### Example 4: PCA with Text Metadata

Reduce dimensionality while excluding text-derived outcome:

```bash
crisp --source ./survey_data \
      --outcome response_category \
      --linkage id \
      --aggregation majority \
      --pca \
      --num 3 \
      --include "q1,q2,q3,q4,q5,q6,q7,q8,q9,q10"
```

## MCP Server Integration

When using CRISP-T with AI assistants via MCP:

### Tool: `regression_analysis`

```json
{
  "name": "regression_analysis",
  "arguments": {
    "outcome": "sentiment_label",
    "include": "age,income,education,satisfaction",
    "linkage_method": "id",
    "aggregation": "mean"
  }
}
```

### Tool: `decision_tree_classification`

```json
{
  "name": "decision_tree_classification",
  "arguments": {
    "outcome": "health_category",
    "top_n": 5,
    "include": "bmi,exercise,diet_score,sleep",
    "linkage_method": "temporal",
    "aggregation": "majority"
  }
}
```

### Tool: `neural_network_classification`

```json
{
  "name": "neural_network_classification",
  "arguments": {
    "outcome": "risk_level",
    "include": "age,weight,blood_pressure,cholesterol",
    "linkage_method": "embedding",
    "aggregation": "majority"
  }
}
```

## Python API

### Direct ML Function Usage

```python
from crisp_t.ml import ML
from crisp_t.csv import Csv
from crisp_t.read_data import ReadData

# Load corpus with both text and numeric data
reader = ReadData()
corpus = reader.from_folder("./data", include_csv=True)

# Assume documents have metadata field 'topic_code' from topic modeling
# and are linked via ID to DataFrame rows

# Initialize ML analyzer
csv_analyzer = Csv(corpus=corpus)
ml_analyzer = ML(csv=csv_analyzer)

# Use text metadata as outcome
result = ml_analyzer.get_regression(
    y="topic_code",  # Text metadata field
    linkage_method="id",
    aggregation="majority"
)

print(f"Model type: {result['model_type']}")
print(f"Accuracy: {result['accuracy']:.2%}")
print("Coefficients:", result['coefficients'])
```

### With Embedding Linkage

```python
from crisp_t.embedding_linker import EmbeddingLinker

# First create embedding links
linker = EmbeddingLinker(corpus)
corpus = linker.link_by_semantic_similarity(
    method="sentence-transformers",
    top_k=1
)

# Save and reload
reader.save_corpus(corpus, "./linked_corpus")
corpus = reader.from_folder("./linked_corpus")

# Now use with ML
csv_analyzer = Csv(corpus=corpus)
ml_analyzer = ML(csv=csv_analyzer)

result = ml_analyzer.get_decision_tree_classes(
    y="sentiment_code",  # From sentiment analysis
    linkage_method="embedding",
    aggregation="majority",
    top_n=10
)
```

## Workflow Example: Complete Pipeline

### Scenario
Analyze patient feedback (text) and health metrics (numeric) to predict satisfaction levels.

### Step 1: Load and Analyze Text

```bash
# Load data and perform sentiment analysis
crisp --source ./patient_feedback \
      --sentiment \
      --out ./analyzed_corpus
```

Documents now have `sentiment_compound` and `sentiment_label` in metadata.

### Step 2: Create Links

```bash
# Link text to numeric data by patient ID
crisp --inp ./analyzed_corpus \
      --source ./patient_metrics.csv \
      --out ./linked_corpus
```

Assuming both have matching 'id' columns, ID linkage is implicit.

### Step 3: ML Prediction

```bash
# Predict sentiment label from health metrics
crisp --inp ./linked_corpus \
      --outcome sentiment_label \
      --linkage id \
      --aggregation majority \
      --regression \
      --include "pain_level,mobility_score,treatment_satisfaction,wait_time" \
      --out ./results
```

### Step 4: Interpret Results

The regression will show:
- Which health metrics predict positive vs negative sentiment
- Coefficient magnitudes indicate importance
- Accuracy shows predictive power

## Best Practices

### 1. Linkage Method Selection
- **Use ID linkage** when you have explicit identifiers
- **Use embedding linkage** for semantic relationships without IDs
- **Use temporal linkage** for time-series data
- **Use keyword linkage** for tag-based categorization

### 2. Aggregation Strategy
- **majority**: Default for classification, handles categorical outcomes
- **mean**: Best for numeric outcomes in regression
- **mode**: Alternative to majority, especially for ordinal data
- **first**: Only when document order has meaning

### 3. Data Preparation
- Ensure text analysis (topics, sentiment, NER) is complete before using metadata
- Verify linkage method creates meaningful connections
- Check that outcome field exists in document metadata
- Validate that linked rows have required numeric features

### 4. Validation
- Always check how many rows were successfully linked
- Review aggregation results when multiple documents link to one row
- Compare results with traditional numeric-only outcomes
- Use cross-validation for model assessment

### 5. Interpretation
- Text metadata outcomes enable qualitative-quantitative triangulation
- Significant predictors suggest numeric patterns underlying qualitative codes
- Low accuracy may indicate:
  - Poor linkage quality
  - Independent qualitative and quantitative dimensions
  - Need for more features or better text analysis

## Troubleshooting

### Error: "No documents with 'X' metadata field"
**Solution**: Run text analysis to generate the metadata field first (e.g., `--sentiment`, `--topics`, `--cat`).

### Error: "No documents linked to DataFrame rows"
**Solution**: 
1. Verify linkage method prerequisites (e.g., embedding_links must exist for embedding linkage)
2. Check that documents have required fields (e.g., timestamp for temporal)
3. Ensure DataFrame has matching column (e.g., 'id' for ID linkage)

### Warning: "Non-numeric values, using majority vote instead"
**Cause**: Aggregation strategy was `mean` but outcome values are non-numeric.
**Solution**: Use `majority` or `mode` for categorical outcomes.

### Low Accuracy Results
**Possible Causes**:
1. Weak relationship between text metadata and numeric features
2. Poor linkage quality (wrong documents linked to rows)
3. Insufficient or irrelevant numeric features
4. Need better text analysis or different metadata field

**Solutions**:
- Try different linkage methods
- Include more relevant numeric features
- Use different text metadata field
- Examine specific misclassified cases

## Advanced: Custom Linkage

For specialized linking needs, you can create custom link metadata:

```python
# Example: Custom keyword linkage
for doc in corpus.documents:
    doc.metadata['keyword_links'] = []
    keywords = doc.metadata.get('extracted_keywords', [])
    for idx, row in corpus.df.iterrows():
        if any(kw in row['tags'] for kw in keywords):
            doc.metadata['keyword_links'].append({
                'df_index': idx,
                'match_score': 1.0
            })

# Save and use
reader.save_corpus(corpus, "./custom_linked")

# Then use with ML
ml_analyzer.get_regression(
    y="category",
    linkage_method="keyword",
    aggregation="majority"
)
```

## Summary

Text metadata outcomes enable powerful triangulation between qualitative and quantitative data:

1. **Linkage Methods**: Connect text to numeric data (id, embedding, temporal, keyword)
2. **Aggregation**: Handle multiple documents per row (majority, mean, first, mode)
3. **ML Tasks**: Use any ML function with text metadata as outcome
4. **Workflows**: CLI, MCP, and Python API all supported
5. **Validation**: Predict qualitative codes from quantitative variables

This feature bridges the gap between qualitative coding and quantitative modeling, enabling comprehensive mixed-methods research within CRISP-T.
