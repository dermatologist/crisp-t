# CSV Query and Correlation Analysis Features

## Overview

CRISP-T now supports advanced CSV data manipulation and analysis capabilities, including:
- Pandas query execution on DataFrames
- Correlation analysis between numeric columns
- Automatic datetime parsing with error handling
- Automatic ID column creation

## Query Execution

### Basic Usage

Execute pandas DataFrame operations directly from the command line:

```bash
# Sort data by a column
crisp --inp ./corpus --query "sort_values('score', ascending=False)"

# Filter rows
crisp --inp ./corpus --query "query('age > 30 and score < 90')"

# Group and aggregate
crisp --inp ./corpus --query "groupby('category')['value'].agg(['mean', 'count', 'std'])"

# Get top N rows
crisp --inp ./corpus --query "nlargest(10, 'score')"
```

### Saving Query Results

By default, query results are displayed but not saved. Use `--save-query-result` to modify the corpus DataFrame:

```bash
# Save the filtered/transformed result back to the corpus
crisp --inp ./corpus --query "query('score > 80')" --save-query-result --out ./filtered_corpus
```

### Query Examples

#### Statistical Aggregations

```bash
# Calculate mean, count, and std deviation by category
crisp --inp ./corpus --query "groupby('topic')['rating'].agg(['mean', 'count', 'std']).sort_values('mean', ascending=False)"
```

Output:
```
           mean  count       std
topic                            
excellent  4.8      15  0.414039
good       4.2      28  0.568331
fair       3.5      12  0.674200
```

#### Filtering and Transformation

```bash
# Complex filtering
crisp --inp ./corpus --query "query('age > 25 and (score > 85 or experience == \"high\")')"

# Select specific columns
crisp --inp ./corpus --query "[['name', 'score', 'category']]"

# Apply transformations
crisp --inp ./corpus --query "assign(score_normalized=lambda x: x['score']/100)"
```

#### Multi-step Queries

For complex analysis, save intermediate results:

```bash
# Step 1: Filter high performers
crisp --inp ./corpus --query "query('score > 90')" --save-query-result --out ./high_performers

# Step 2: Analyze high performers
crisp --inp ./high_performers --query "groupby('department')['salary'].mean()"
```

## Correlation Analysis

### Basic Usage

Compute correlation matrices and identify significant relationships:

```bash
# Default: Pearson correlation with threshold 0.5
crisp --inp ./corpus --correlation

# Custom threshold
crisp --inp ./corpus --correlation --correlation-threshold 0.7

# Different correlation method
crisp --inp ./corpus --correlation --correlation-method spearman
```

### Output

The correlation analysis shows:
1. **Significant Correlations**: Pairs of variables with correlation above the threshold
2. **Full Correlation Matrix**: Complete correlation between all numeric columns

Example output:
```
╔═══════════════════════════════════════╗
║         📈  CORRELATION ANALYSIS         ║
╚═══════════════════════════════════════╝

🔗 Significant Correlations (threshold=0.7, method=pearson):
   Variable 1  Variable 2  Correlation
0        age          score     0.892341
1     experience     salary     0.845123

📊 Full Correlation Matrix:
              age     score  experience    salary
age          1.000     0.892       0.623     0.534
score        0.892     1.000       0.701     0.612
experience   0.623     0.701       1.000     0.845
salary       0.534     0.612       0.845     1.000
```

### Correlation Methods

Three methods are supported:

1. **Pearson** (default): Measures linear relationships
   ```bash
   crisp --inp ./corpus --correlation --correlation-method pearson
   ```

2. **Spearman**: Measures monotonic relationships (rank-based)
   ```bash
   crisp --inp ./corpus --correlation --correlation-method spearman
   ```

3. **Kendall**: Measures ordinal associations
   ```bash
   crisp --inp ./corpus --correlation --correlation-method kendall
   ```

### Use Cases

#### Identifying Multicollinearity

Before building predictive models:
```bash
# Check for highly correlated predictors
crisp --inp ./corpus --correlation --correlation-threshold 0.9
```

#### Feature Selection

Find variables strongly correlated with your outcome:
```bash
# Load data and filter to include outcome variable
crisp --inp ./corpus --query "[['outcome', 'feature1', 'feature2', 'feature3']]" --correlation --correlation-threshold 0.5
```

#### Exploratory Data Analysis

Discover unexpected relationships:
```bash
# Lower threshold to explore all relationships
crisp --inp ./corpus --correlation --correlation-threshold 0.3
```

## Automatic Datetime Parsing

When importing CSV files, date columns are automatically detected and parsed:

```bash
# Dates in columns will be automatically parsed
crisp --source ./data --out ./corpus
```

Features:
- Detects columns with date-like content (>50% valid dates)
- Uses `errors='coerce'` to handle invalid dates gracefully
- Invalid dates become NaT (Not-a-Time) instead of causing errors
- Supports various date formats (ISO 8601, MM/DD/YYYY, etc.)

Example CSV:
```csv
id,event_date,score
1,2025-01-15,85
2,2025-02-20,92
3,invalid_date,78
```

The `event_date` column will be parsed as datetime, with the invalid date becoming NaT.

## Automatic ID Column Creation

If you specify an ID column that doesn't exist, it will be created from the DataFrame index:

```bash
# patient_id doesn't exist in CSV - will be created from index
crisp --source ./data -t notes --out ./corpus
```

This ensures every row has a unique identifier for linking purposes.

## Workflow Examples

### Example 1: Survey Data Analysis

```bash
# 1. Import survey data with multiple text columns
crisp --source ./survey_data \
      -t "question1,question2,question3" \
      --out ./survey_corpus

# 2. Check correlations between numeric ratings
crisp --inp ./survey_corpus \
      --correlation \
      --correlation-threshold 0.6

# 3. Filter high satisfaction responses
crisp --inp ./survey_corpus \
      --query "query('satisfaction > 4')" \
      --save-query-result \
      --out ./high_satisfaction

# 4. Analyze text from high satisfaction group
crisp --inp ./high_satisfaction --topics --sentiment
```

### Example 2: Clinical Research Data

```bash
# 1. Import patient records
crisp --source ./patient_data \
      -t "symptoms,history,notes" \
      --out ./clinical_corpus

# 2. Find patients with similar profiles (high correlation metrics)
crisp --inp ./clinical_corpus \
      --correlation \
      --correlation-method spearman

# 3. Group by diagnosis and calculate statistics
crisp --inp ./clinical_corpus \
      --query "groupby('diagnosis')[['age', 'severity', 'duration']].agg(['mean', 'std', 'count'])"

# 4. Filter and analyze specific patient group
crisp --inp ./clinical_corpus \
      --query "query('age > 50 and severity > 7')" \
      --save-query-result \
      --out ./high_risk_patients

crisp --inp ./high_risk_patients --codedict --summary
```

### Example 3: Longitudinal Study

```bash
# 1. Import time-series data (dates will be auto-parsed)
crisp --source ./longitudinal_data \
      -t "observations,notes" \
      --out ./time_corpus

# 2. Calculate change over time
crisp --inp ./time_corpus \
      --query "sort_values('assessment_date').diff()"

# 3. Correlate changes across measures
crisp --inp ./time_corpus \
      --correlation \
      --correlation-threshold 0.5 \
      --correlation-method pearson
```

## Programmatic Usage

For use in Python scripts or Jupyter notebooks:

```python
from crisp_t.csv import Csv
from crisp_t.read_data import ReadData

# Load corpus
reader = ReadData()
corpus = reader.read_corpus_from_json('./corpus')

# Create CSV analyzer
csv_analyzer = Csv(corpus=corpus)

# Execute query
result = csv_analyzer.execute_query("groupby('category')['value'].mean()")
print(result)

# Compute correlations
corr_matrix = csv_analyzer.compute_correlation(threshold=0.5, method='pearson')
print(corr_matrix)

# Find significant correlations
sig_corrs = csv_analyzer.find_significant_correlations(threshold=0.7)
print(sig_corrs)
```

## Tips and Best Practices

1. **Start with Exploration**: Use correlation analysis before query execution to understand data relationships
2. **Use Appropriate Thresholds**: 
   - 0.3-0.5: Weak correlation
   - 0.5-0.7: Moderate correlation
   - 0.7-0.9: Strong correlation
   - 0.9+: Very strong correlation (check for multicollinearity)
3. **Save Intermediate Results**: Use `--save-query-result --out` to preserve filtered/transformed data
4. **Validate Queries**: Test complex queries on small datasets first
5. **Choose Correlation Method**:
   - Pearson: For linear relationships and normally distributed data
   - Spearman: For monotonic relationships and ordinal data
   - Kendall: For small datasets and ordinal data

## Error Handling

### Invalid Queries

```bash
# This will fail with a descriptive error
crisp --inp ./corpus --query "nonexistent_method()"
# Error: Invalid query: 'DataFrame' object has no attribute 'nonexistent_method'
```

### Empty Results

```bash
# If no correlations found
crisp --inp ./corpus --correlation --correlation-threshold 0.99
# Output: No significant correlations found above threshold 0.99
```

### Missing Columns

```bash
# If querying nonexistent column
crisp --inp ./corpus --query "query('nonexistent_col > 5')"
# Error: Invalid query: 'nonexistent_col'
```

## Related Documentation

- [Multiple Text Columns](./MULTIPLE_TEXT_COLUMNS_EXAMPLE.md) - Importing multiple CSV text columns
- [Cheatsheet](../docs/cheatsheet.md) - Quick reference for all CLI commands
- [DEMO.md](../docs/DEMO.md) - Step-by-step walkthrough
