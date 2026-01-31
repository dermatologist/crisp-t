# Importing Multiple Text Columns from CSV

## Overview

CRISP-T supports importing multiple text columns from CSV files, allowing you to combine multiple text fields into a single document for analysis. This is particularly useful when working with survey data, medical records, or any dataset where information is spread across multiple text fields.

## Basic Usage

There are two ways to specify multiple text columns when importing CSV data:

### Method 1: Multiple `-t` Flags

Use the `-t` (or `--unstructured`) flag multiple times to specify each column:

```bash
crisp --source ./survey_data -t response1 -t response2 -t response3 --out ./corpus
```

### Method 2: Comma-Separated Column Names

Provide a comma-separated list of column names in a single `-t` flag:

```bash
crisp --source ./survey_data -t "response1,response2,response3" --out ./corpus
```

Both methods produce the same result - the text from all specified columns will be concatenated into a single document for each row.

## Example Scenario: Patient Survey Data

Suppose you have a CSV file (`patient_survey.csv`) with the following structure:

```csv
patient_id,chief_complaint,history_present_illness,review_of_systems,satisfaction_score
1001,Headache,Severe headaches for 2 weeks,No other symptoms reported,4
1002,Chest pain,Intermittent chest pain,Also experiencing shortness of breath,3
1003,Fatigue,Feeling tired all day,No appetite and weight loss,5
```

### Import All Text Columns

To analyze all three text columns together:

```bash
# Create a directory and copy your CSV there
mkdir -p ./patient_data
cp patient_survey.csv ./patient_data/

# Import with multiple text columns
crisp --source ./patient_data \
      -t chief_complaint \
      -t history_present_illness \
      -t review_of_systems \
      --out ./patient_corpus
```

Or using the comma-separated approach:

```bash
crisp --source ./patient_data \
      -t "chief_complaint,history_present_illness,review_of_systems" \
      --out ./patient_corpus
```

### Result

Each row in the CSV becomes a single document with text concatenated from all three columns:

- **Document 1**: "Headache Severe headaches for 2 weeks No other symptoms reported"
- **Document 2**: "Chest pain Intermittent chest pain Also experiencing shortness of breath"
- **Document 3**: "Fatigue Feeling tired all day No appetite and weight loss"

The numeric columns (`patient_id` and `satisfaction_score`) remain in the DataFrame for quantitative analysis.

## Advanced Usage

### Combining with Other Options

#### Limit Number of Rows

Import only the first 100 rows:

```bash
crisp --source ./survey_data \
      -t "response1,response2,response3" \
      --rec 100 \
      --out ./corpus
```

#### Ignore Stop Words

Remove common words from all text columns:

```bash
crisp --source ./survey_data \
      -t "response1,response2,response3" \
      --ignore "the,and,or,but,is,are" \
      --out ./corpus
```

#### Specify ID Column

Use a specific column as the document ID:

```bash
crisp --source ./survey_data \
      -t "response1,response2,response3" \
      --out ./corpus

# Then when analyzing, use ID linking
crispt --inp ./corpus --add-rel "text:symptom|numb:severity_score|predicts"
```

## Common Use Cases

### 1. Medical Notes

Combine multiple sections of medical documentation:

```bash
crisp --source ./medical_records \
      -t "chief_complaint,history,physical_exam,assessment,plan" \
      --out ./medical_corpus
```

### 2. Survey Responses

Merge multiple open-ended survey questions:

```bash
crisp --source ./survey_results \
      -t "question1,question2,question3,comments" \
      --out ./survey_corpus
```

### 3. Social Media Data

Combine post text, comments, and replies:

```bash
crisp --source ./social_data \
      -t "post_text,top_comment,user_reply" \
      --out ./social_corpus
```

## Subsequent Analysis

After importing, you can perform various analyses on the combined text:

### Topic Modeling

```bash
crisp --inp ./patient_corpus --topics --num 5 --out ./analyzed_corpus
```

### Sentiment Analysis

```bash
crisp --inp ./patient_corpus --sentiment --out ./analyzed_corpus
```

### Linking Text to Numeric Data

```bash
# Link by keywords (default)
crisp --inp ./analyzed_corpus --outcome satisfaction_score --regression

# Link by ID
crisp --inp ./analyzed_corpus --linkage id --outcome satisfaction_score --cls
```

### Visualizations

```bash
crispviz --inp ./analyzed_corpus --wordcloud --ldavis --out ./visualizations
```

## Tips and Best Practices

1. **Column Order Matters**: Text is concatenated in the order you specify the columns. List the most important columns first if order is relevant for your analysis.

2. **Handling Missing Values**: Empty or missing values in any column are handled gracefully - they won't appear as "NaN" or "None" in your documents.

3. **Whitespace in Column Names**: Spaces around column names in comma-separated lists are automatically stripped, so `"col1, col2, col3"` works the same as `"col1,col2,col3"`.

4. **Nonexistent Columns**: If you specify a column that doesn't exist in your CSV, it will be silently skipped. This allows for flexible scripts that work with different CSV formats.

5. **Text Column Removal**: After importing, the specified text columns are removed from the DataFrame, leaving only numeric and categorical columns for quantitative analysis.

6. **Backward Compatibility**: Single column import still works as before:
   ```bash
   crisp --source ./data -t comments --out ./corpus
   ```

## Troubleshooting

### Problem: Text from some columns is missing

**Solution**: Check that column names match exactly (case-sensitive) with your CSV headers. Use:

```bash
# View actual column names in your CSV
head -1 your_file.csv
```

### Problem: Getting warnings about columns not found in DataFrame

**Solution**: This is expected after import - text columns are removed from the DataFrame since they're now in the documents. These warnings can be ignored.

### Problem: Documents appear empty or incomplete

**Solution**: Verify your CSV file encoding is UTF-8, and check for special characters that might cause parsing issues.

## Related Documentation

- [Cheatsheet](../docs/cheatsheet.md) - Quick reference for all CLI commands
- [DEMO.md](../docs/DEMO.md) - Step-by-step walkthrough
- [INSTRUCTION.md](./INSTRUCTION.md) - Comprehensive user guide
