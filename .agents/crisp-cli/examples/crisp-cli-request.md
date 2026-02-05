# CRISP-T CLI Agent - Example Requests

This document shows example requests an agent might handle when using CRISP-T CLI tools.

## Basic Examples

### Example 1: Import and Explore Data

**User Request**: "I have interview data in a folder. Can you import it and show me what we're working with?"

**Agent Response and Commands**:
```bash
# Step 1: Import the data
crisp --source interviews_folder \
      --unstructured "responses,comments" \
      --out interview_corpus \
      --num 100 \
      --verbose

# Step 2: Explore the structure
crispt --inp interview_corpus --print
crispt --inp interview_corpus --doc-ids
crispt --inp interview_corpus --df-cols
crispt --inp interview_corpus --df-row-count
```

**Expected Output**:
- Corpus created with N documents
- List of document IDs
- DataFrame columns and row count
- Overall corpus metadata

---

### Example 2: Quick Text Analysis

**User Request**: "What are the main topics and sentiment in our interview data?"

**Agent Commands**:
```bash
# Run comprehensive text analysis
crisp --inp interview_corpus \
      --codedict \
      --topics --num 5 \
      --assign \
      --sentiment \
      --out interview_corpus_v1 \
      --verbose

# Visualize results
crispviz --inp interview_corpus_v1 \
         --out visualizations \
         --freq \
         --wordcloud \
         --by-topic \
         --top-n 20
```

**Expected Output**:
- Coding dictionary (verbs, nouns, adjectives)
- 5 topics with keywords
- Document-topic assignments
- Sentiment scores
- Word frequency chart
- Word clouds for each topic
- Topic distribution visualization

---

### Example 3: Mixed-Methods Analysis

**User Request**: "Link our interview themes to survey outcomes. Do healthcare concerns correlate with satisfaction scores?"

**Agent Commands**:
```bash
# 1. Analyze text
crisp --inp interview_corpus \
      --topics --num 4 \
      --assign \
      --out corpus_topics

# 2. Create relationships linking themes to outcomes
crispt --inp corpus_topics \
       --add-rel "text:healthcare|num:satisfaction_score|correlates" \
       --add-rel "text:cost_barriers|num:willingness_to_pay|predicts" \
       --add-rel "text:access_issues|num:appointment_frequency|predicts" \
       --out corpus_linked

# 3. Validate with regression
crisp --inp corpus_linked \
      --regression \
      --outcome satisfaction_score \
      --linkage keyword \
      --aggregation mean \
      --out regression_results

# 4. Visualize relationships
crispviz --inp regression_results \
         --out viz \
         --graph \
         --corr-heatmap \
         --wordcloud

# 5. Display relationships
crispt --inp regression_results --print-relationships
```

**Expected Output**:
- Regression coefficients showing correlation strength
- Network graph of text↔numeric connections
- Correlation heatmap
- Confirmed relationships format: `text:keyword|num:column|relation`

---

### Example 4: Filtering and Subsetting

**User Request**: "I only want to analyze responses from the North region. What are the themes?"

**Agent Commands**:
```bash
# Filter and analyze subset
crisp --inp interview_corpus \
      --filters "region=North" \
      --topics --num 4 \
      --assign \
      --sentiment \
      --out north_region_analysis

# Compare to South
crisp --inp interview_corpus \
      --filters "region=South" \
      --topics --num 4 \
      --assign \
      --out south_region_analysis

# Visualize both
crispviz --inp north_region_analysis --out viz_north --wordcloud
crispviz --inp south_region_analysis --out viz_south --wordcloud
```

**Expected Output**:
- Separate corpora for each region
- Region-specific topics
- Visualizations showing regional differences

---

### Example 5: Semantic Search (Literature Review)

**User Request**: "Find all documents similar to interview_5 which discusses healthcare barriers."

**Agent Commands**:
```bash
# Find similar documents
crispt --inp interview_corpus \
       --similar-docs "interview_5" \
       --num 10 \
       --rec 0.7

# Perform semantic search on a concept
crispt --inp interview_corpus \
       --semantic "healthcare access barriers" \
       --num 15

# Find chunks within interview_5 mentioning cost
crispt --inp interview_corpus \
       --semantic-chunks "cost barriers" \
       --doc-id interview_5 \
       --rec 0.5
```

**Expected Output**:
- List of 10 documents similar to interview_5 (sorted by similarity)
- 15 documents containing "healthcare access barriers" concept
- Relevant passages from interview_5 about cost

---

## Advanced Examples

### Example 6: Machine Learning Classification

**User Request**: "Train a model to predict satisfaction from our interview themes and demographic data."

**Agent Commands**:
```bash
# Prepare corpus with text features
crisp --inp interview_corpus \
      --topics --num 5 \
      --assign \
      --out corpus_features

# Train classifier
crisp --inp corpus_features \
      --cls \
      --outcome satisfaction_category \
      --include age,income,years_experience \
      --linkage keyword \
      --aggregation majority \
      --out classification_results

# View feature importance
crispt --inp classification_results --print-relationships

# Visualize predictions
crispviz --inp classification_results \
         --out viz \
         --graph \
         --corr-heatmap
```

**Expected Output**:
- Feature importance rankings
- Relationships showing which text themes predict satisfaction
- Confusion matrix for classification accuracy
- Network visualization of predictions

---

### Example 7: Temporal Analysis

**User Request**: "How has sentiment about healthcare changed over time? Link our time-series interview dates to our weekly survey responses."

**Agent Commands**:
```bash
# Link documents by time
crispt --inp interview_corpus \
       --temporal-link "sequence:date:W" \
       --out corpus_temporal

# Filter to recent period
crispt --inp corpus_temporal \
       --temporal-filter "2025-01-01:2025-12-31" \
       --out corpus_recent

# Analyze sentiment over time
crisp --inp corpus_recent \
      --sentiment \
      --sentence \
      --out temporal_sentiment_results

# Generate weekly summary
crispt --inp temporal_sentiment_results --temporal-summary "W"

# Visualize trends
crispviz --inp temporal_sentiment_results \
         --out viz_temporal \
         --freq \
         --by-topic
```

**Expected Output**:
- Documents linked to weekly time periods
- Sentiment scores by week
- Visualizations showing sentiment trends
- Temporal summary statistics

---

### Example 8: Complex Pipeline

**User Request**: "Run a complete analysis: import data, perform text analysis, validate with ML, and generate visualizations."

**Agent Commands**:
```bash
# 1. Import
crisp --source raw_data_folder \
      --unstructured responses,notes \
      --out corpus_v0 \
      --num 200

# 2. Text analysis
crisp --inp corpus_v0 \
      --codedict \
      --topics --num 6 \
      --assign \
      --sentiment \
      --out corpus_v1

# 3. Category extraction
crisp --inp corpus_v1 \
      --cat \
      --out corpus_v2

# 4. Create relationships
crispt --inp corpus_v2 \
       --add-rel "text:cost_barriers|num:household_income|correlates" \
       --add-rel "text:access_issues|num:visit_frequency|predicts" \
       --add-rel "text:satisfaction|num:nps_score|predicts" \
       --out corpus_v3

# 5. ML validation
crisp --inp corpus_v3 \
      --regression \
      --outcome nps_score \
      --linkage keyword \
      --aggregation mean \
      --include age,region \
      --out results_final

# 6. Comprehensive visualization
crispviz --inp results_final \
         --out final_visualizations \
         --freq \
         --top-terms \
         --wordcloud \
         --by-topic \
         --corr-heatmap \
         --graph \
         --graph-nodes document,keyword

# 7. Final report
crispt --inp results_final --print-relationships
crispt --inp results_final --print
```

**Expected Output**:
- Complete analyzed corpus
- All relationships documented
- ML coefficients linking text to outcomes
- Publication-ready visualizations
- Comprehensive corpus summary

---

## Edge Case Examples

### Example 9: Handling Missing Linkage

**Scenario**: User wants ML with text outcome but forgets `--linkage`

**Bad Request**:
```bash
crisp --inp corpus --regression --outcome interview_theme --out results
# Error: outcome not found in DataFrame
```

**Corrected Request**:
```bash
crisp --inp corpus --regression --outcome interview_theme \
      --linkage keyword --aggregation majority --out results
# Success: Uses text metadata for outcome
```

---

### Example 10: Cache Management

**Scenario**: User wants to re-run analysis with different filters

**Wrong Approach**:
```bash
crisp --inp corpus --assign --out results
# Cache error: existing cache from previous run
```

**Correct Approach**:
```bash
crisp --clear --inp corpus --assign --out results
# Success: Cache cleared, fresh assignment
```

---

### Example 11: Performance Optimization

**Scenario**: Working with 50,000 documents

**Inefficient**:
```bash
crisp --source huge_dataset --out corpus  # Takes forever
```

**Efficient**:
```bash
# Step 1: Sample during import
crisp --source huge_dataset --out corpus --num 1000 --rec 100

# Step 2: Analyze sample
crisp --inp corpus --nlp --out sample_analysis

# Step 3: Later, expand if needed
crisp --source huge_dataset --out corpus_full --num 10000
```

---

## Request Patterns for Agents

### Pattern 1: Exploratory Analysis
```
User: "What's in our data?"
Agent:
  1. Load corpus
  2. Display structure
  3. Generate coding dictionary
  4. Show quick statistics
  5. Create visualizations
```

### Pattern 2: Validation
```
User: "Does X relate to Y?"
Agent:
  1. Analyze X (text topics)
  2. Create relationships X→Y
  3. Run regression/correlation
  4. Visualize relationships
  5. Report coefficients
```

### Pattern 3: Comparison
```
User: "How does Group A differ from Group B?"
Agent:
  1. Filter to Group A
  2. Filter to Group B
  3. Analyze each separately
  4. Compare results
  5. Visualize differences
```

### Pattern 4: Publication
```
User: "Prepare data for paper"
Agent:
  1. Complete analysis
  2. Create relationships
  3. Generate publication visualizations
  4. Export metadata/results
  5. Generate summary report
```

---

## Notes for Agents

1. **Always use `--out`** to save intermediate results
2. **Use `--clear`** before `--assign` if filters/data changed
3. **Specify `--linkage`** when outcome is text field
4. **Use `--unstructured`** to mark free-text CSV columns
5. **Check metadata** with `crispt --print` before complex operations
6. **Start small** (use `--num` to limit) during development
7. **Combine flags** (`--nlp` = codedict + topics + sentiment) for efficiency
8. **Validate results** by checking relationships and visualizations
9. **Document workflows** by showing commands to user
10. **Handle errors** gracefully and suggest corrections

