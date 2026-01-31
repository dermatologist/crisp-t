# Advanced Data Analysis with CRISP-T MCP Tools

This guide demonstrates the new data analysis tools added to CRISP-T's MCP server, showcasing advanced DataFrame operations, correlation analysis, and dynamic queries.

## Overview of New Tools

### Data Cleaning Tools (DataFrame Operations)
- `get_df_shape` - Get DataFrame dimensions
- `mark_missing` - Remove rows with empty/NaN values
- `mark_duplicates` - Remove duplicate rows  
- `restore_df` - Restore DataFrame to original state
- `drop_na` - Remove all rows with any NA values

### Advanced Encoding Tools (Column Operations)
- `one_hot_encode_strings_in_df` - Batch encode object columns with cardinality filtering
- `one_hot_encode_all_columns` - Convert all to boolean for association rules

### Statistical Analysis Tools (Data Analysis)
- `compute_correlation` - Compute correlation matrix
- `find_significant_correlations` - Find strong correlations
- `execute_query` - Execute dynamic pandas queries
- `get_column_statistics` - Get statistical summaries
- `get_unique_values_summary` - Analyze unique values and cardinality

## Example Workflows

### Workflow 1: Data Quality Assessment

```python
# Step 1: Load your corpus with CSV data
load_corpus(inp="path/to/corpus.json")

# Step 2: Check DataFrame dimensions
get_df_shape()
# Output: "DataFrame shape: 1000 rows × 25 columns"

# Step 3: Get column information
get_df_columns()

# Step 4: Analyze data quality
get_unique_values_summary(columns="status,category,region")
# Shows: unique counts, top values, cardinality for each column

# Step 5: Get statistical summary for numeric columns
get_column_statistics(columns="age,score,income")
# Shows: count, mean, std, min, quartiles, max

# Step 6: Clean the data
mark_missing()  # Remove rows with empty strings or NaN
mark_duplicates()  # Remove duplicate rows

# Step 7: Check shape after cleaning
get_df_shape()
# Output: "DataFrame shape: 950 rows × 25 columns"
```

### Workflow 2: Correlation Analysis

```python
# Step 1: Load corpus with numeric data
load_corpus(source="data/survey_results.csv", text_columns="comments")

# Step 2: Compute full correlation matrix
compute_correlation(method="pearson")
# Shows correlation matrix for all numeric columns

# Step 3: Find significant correlations only
find_significant_correlations(threshold=0.7, method="pearson")
# Output:
# Significant Correlations (|r| >= 0.7, pearson method):
# Found 3 significant correlation(s):
#   satisfaction ↔ recommend_score: 0.856
#   price_perception ↔ value_score: 0.782
#   service_quality ↔ overall_rating: 0.734

# Step 4: Investigate specific variables
compute_correlation(
    columns="satisfaction,price,quality,service",
    method="spearman",  # Use Spearman for non-linear relationships
    threshold=0.5
)

# Step 5: Document findings
add_relationship(
    first="num:satisfaction",
    second="num:recommend_score",
    relation="strongly_correlates"
)
```

### Workflow 3: Dynamic Data Queries

```python
# Step 1: Load data
load_corpus(source="data/customer_data.csv")

# Step 2: Filter data with queries
execute_query(
    query="age > 25 and status == 'active'",
    save_result=True
)
# Filters DataFrame to active customers over 25

# Step 3: Group and aggregate
execute_query(
    query="groupby:department agg:mean",
    save_result=True
)
# Groups by department and shows mean values

# Step 4: Sort results
execute_query(
    query="sort:salary:desc",
    save_result=False  # Preview without saving
)
# Sorts by salary descending

# Step 5: Get top records
execute_query(
    query="head:20",
    save_result=True
)
# Keep only top 20 records
```

### Workflow 4: Advanced Feature Engineering

```python
# Step 1: Load data
load_corpus(source="data/mixed_types.csv")

# Step 2: Inspect data types
get_column_types()

# Step 3: Get cardinality summary
get_unique_values_summary(top_n=5)
# Shows which columns are high-cardinality (>100 unique values)

# Step 4: Batch encode all categorical columns
one_hot_encode_strings_in_df(
    n=10,  # Top 10 categories per column
    filter_high_cardinality=True  # Skip columns with >100 unique values
)
# Automatically handles multiple categorical columns

# Step 5: Verify the encoding worked
get_df_columns()  # Shows new one-hot encoded columns
get_column_types()  # All should be numeric now

# Step 6: Proceed to ML analysis
decision_tree_classification(
    outcome="target",
    include="all_encoded_features",
    top_n=15
)
```

### Workflow 5: Association Rules Mining

```python
# Step 1: Load transactional data
load_corpus(source="data/transactions.csv")

# Step 2: Check data
get_df_shape()
get_unique_values_summary()

# Step 3: Convert to boolean encoding for association rules
one_hot_encode_all_columns()
# Converts all values to True/False

# Step 4: Run association rules analysis
association_rules(
    outcome="purchased",
    include="product_a,product_b,product_c,product_d",
    min_support=50,
    min_threshold=70
)
# Finds patterns like: "If customer bought A and B, then likely bought C"

# Step 5: If needed, restore original data
restore_df()
```

### Workflow 6: Handling Missing Data

```python
# Step 1: Load data
load_corpus(source="data/incomplete_data.csv")

# Step 2: Check initial state
get_df_shape()
# Output: "DataFrame shape: 1500 rows × 30 columns"

# Step 3: Check which columns have missing values
get_column_statistics()  # count shows non-null values per column

# Step 4: Option A - Remove any row with missing data
drop_na()  # Strict: removes rows with ANY NA
get_df_shape()
# Output: "DataFrame shape: 850 rows × 30 columns"

# If too aggressive, restore and try gentler approach
restore_df()

# Step 5: Option B - Remove only rows with empty strings
mark_missing()  # Gentler: only removes empty strings and explicit NaN
get_df_shape()
# Output: "DataFrame shape: 1200 rows × 30 columns"

# Step 6: Proceed with analysis on clean data
```

### Workflow 7: Mixed-Methods Research

```python
# Step 1: Load corpus with both text and numeric data
load_corpus(
    source="data/interview_study.csv",
    text_columns="interview_transcript,comments"
)

# Step 2: Analyze text
topic_modeling(num_topics=5, num_words=10)
assign_topics(num_topics=5)
sentiment_analysis(documents=True)

# Step 3: Analyze correlations in numeric data
find_significant_correlations(
    threshold=0.6,
    method="spearman"
)

# Step 4: Check unique values in categorical variables
get_unique_values_summary(
    columns="gender,education,occupation",
    top_n=10
)

# Step 5: Filter to specific subgroup for deeper analysis
execute_query(
    query="education == 'Graduate' and age > 30",
    save_result=True
)

# Step 6: Re-analyze text in subgroup
topic_modeling(num_topics=3, num_words=8)

# Step 7: Compare with numeric patterns
compute_correlation(
    columns="satisfaction,engagement,outcome",
    method="pearson"
)

# Step 8: Document text-numeric relationships
add_relationship(
    first="text:technology_theme",
    second="num:satisfaction",
    relation="positively_correlates"
)
```

### Workflow 8: Iterative Data Exploration

```python
# Step 1: Load data
load_corpus(source="data/complex_dataset.csv")

# Step 2: Get overview
get_df_shape()
get_column_statistics()

# Step 3: Try a transformation
bin_a_column(column_name="age", bins=5)

# Step 4: Check result
get_column_values(column_name="age")

# Step 5: Not satisfied? Restore and try different binning
restore_df()
bin_a_column(column_name="age", bins=3)

# Step 6: Better! Now encode it
one_hot_encode_column(column_name="age")

# Step 7: Verify
get_df_columns()  # See new columns: age_low, age_medium, age_high

# Step 8: Save progress
save_corpus(out="data/processed_corpus.json")
```

## Best Practices

### Data Cleaning
1. Always check `get_df_shape()` before and after cleaning
2. Use `restore_df()` if cleaning is too aggressive
3. `mark_missing()` is gentler than `drop_na()`
4. Remove duplicates early with `mark_duplicates()`

### Correlation Analysis
1. Start with `find_significant_correlations()` to get overview
2. Use `threshold=0.5` for moderate correlations, 0.7+ for strong
3. Try different methods: Pearson (linear), Spearman (monotonic), Kendall (ordinal)
4. Document significant correlations with `add_relationship()`

### Feature Engineering
1. Use `get_unique_values_summary()` to understand cardinality
2. `one_hot_encode_strings_in_df()` for batch encoding
3. Set `filter_high_cardinality=True` to avoid explosion
4. Verify with `get_column_types()` after encoding

### Dynamic Queries
1. Test queries with `save_result=False` first
2. Use `execute_query()` for complex filters that need multiple conditions
3. Combine with traditional tools for complete workflows
4. Remember: queries modify the DataFrame if `save_result=True`

### Data Restoration
1. `restore_df()` returns to original loaded state
2. Use liberally during exploration
3. Cannot undo a restore (no redo functionality)
4. Save good intermediate states with `save_corpus()`

## Integration with Existing Tools

The new tools complement existing CRISP-T features:

**Text Analysis**
- Use `find_significant_correlations()` after `sentiment_analysis()` to see if sentiment correlates with outcomes
- Filter data with `execute_query()` then run `topic_modeling()` on subsets

**ML Analysis**
- `get_column_statistics()` before feature selection
- `compute_correlation()` to detect multicollinearity before regression
- `one_hot_encode_strings_in_df()` for automated preprocessing

**Temporal Analysis**
- `execute_query()` to filter time periods
- `find_significant_correlations()` to detect trends over time

**Mixed Methods**
- `get_unique_values_summary()` to understand categorical patterns
- Link findings with `add_relationship()` 
- Use `compute_correlation()` to validate qualitative themes

## Tips and Tricks

1. **Quick Data Check**: `get_df_shape()` + `get_column_statistics()` + `get_unique_values_summary()`
2. **Find Patterns**: `find_significant_correlations(threshold=0.7)` 
3. **Clean Pipeline**: `mark_missing()` → `mark_duplicates()` → `get_df_shape()`
4. **Safe Exploration**: Make changes → Check results → If bad, `restore_df()`
5. **Batch Encoding**: `one_hot_encode_strings_in_df(n=10, filter_high_cardinality=True)`

## Conclusion

These 12 new tools extend CRISP-T's data analysis capabilities, making it easier to:
- Clean and prepare data
- Discover statistical relationships
- Perform complex data manipulations
- Explore data interactively
- Integrate numeric and qualitative analysis

All tools work seamlessly with existing CRISP-T features for comprehensive mixed-methods research.
