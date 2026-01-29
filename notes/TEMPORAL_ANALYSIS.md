# Temporal Analysis Guide for CRISP-T

## Overview

CRISP-T now supports comprehensive time-based analysis capabilities, allowing you to:
- Link text documents to numeric data based on timestamps
- Filter and analyze data by time ranges
- Track sentiment and topics over time
- Generate temporal graphs and visualizations
- Perform time-series triangulation of qualitative and quantitative data

## Getting Started

### Prerequisites

- Documents or CSV data with timestamp information
- Timestamps should be in ISO 8601 format (e.g., `2025-01-15T10:30:00`) or common date formats

### Adding Timestamps to Documents

When creating documents programmatically:

```python
from crisp_t import Document

doc = Document(
    id="doc1",
    name="Patient Note",
    text="Patient reports improvement in symptoms",
    timestamp="2025-01-15T10:30:00"
)
```

### CSV Data with Timestamps

CRISP-T automatically detects timestamp columns named:
- `timestamp`
- `datetime`
- `time`
- `date`
- `created_at`
- `updated_at`

Example CSV:
```csv
id,timestamp,value,category
1,2025-01-15T10:00:00,100,A
2,2025-01-15T11:00:00,150,B
3,2025-01-15T12:00:00,120,A
```

## Core Temporal Functions

### 1. Temporal Linking

Link documents to numeric data based on timestamps.

#### Nearest Time Linking

Links each document to the nearest dataframe row in time:

```python
from crisp_t import TemporalAnalyzer

analyzer = TemporalAnalyzer(corpus)
corpus = analyzer.link_by_nearest_time(time_column="timestamp")

# With maximum time gap constraint (e.g., 10 minutes)
from datetime import timedelta
corpus = analyzer.link_by_nearest_time(
    time_column="timestamp",
    max_gap=timedelta(minutes=10)
)
```

**CLI Usage:**
```bash
crispt --inp corpus_folder --temporal-link "nearest:timestamp"
```

#### Time Window Linking

Links documents to all dataframe rows within a time window:

```python
corpus = analyzer.link_by_time_window(
    time_column="timestamp",
    window_before=timedelta(minutes=5),
    window_after=timedelta(minutes=5)
)
```

**CLI Usage:**
```bash
# ±300 seconds (5 minutes) window
crispt --inp corpus_folder --temporal-link "window:timestamp:300"
```

#### Sequence-Based Linking

Links documents and data by time periods (day, week, month):

```python
# Link by week
corpus = analyzer.link_by_sequence(time_column="timestamp", period="W")

# Link by day
corpus = analyzer.link_by_sequence(time_column="timestamp", period="D")

# Link by month
corpus = analyzer.link_by_sequence(time_column="timestamp", period="M")
```

**CLI Usage:**
```bash
crispt --inp corpus_folder --temporal-link "sequence:timestamp:W"
```

### 2. Temporal Filtering

Filter corpus by time range:

```python
# Filter to specific date range
filtered_corpus = analyzer.filter_by_time_range(
    start_time="2025-01-01T00:00:00",
    end_time="2025-06-30T23:59:59"
)

# Filter from start date onwards
filtered_corpus = analyzer.filter_by_time_range(
    start_time="2025-01-01T00:00:00",
    end_time=None
)

# Filter up to end date
filtered_corpus = analyzer.filter_by_time_range(
    start_time=None,
    end_time="2025-12-31T23:59:59"
)
```

**CLI Usage:**
```bash
# Filter to specific range
crispt --inp corpus_folder --temporal-filter "2025-01-01:2025-06-30"

# From date onwards
crispt --inp corpus_folder --temporal-filter "2025-01-01:"

# Up to date
crispt --inp corpus_folder --temporal-filter ":2025-06-30"
```

### 3. Temporal Summaries

Generate summaries of data over time periods:

```python
# Weekly summary
summary_df = analyzer.get_temporal_summary(
    time_column="timestamp",
    period="W"
)

# Daily summary with specific columns
summary_df = analyzer.get_temporal_summary(
    time_column="timestamp",
    period="D",
    numeric_columns=["value", "score"]
)
```

**CLI Usage:**
```bash
# Weekly summary
crispt --inp corpus_folder --temporal-summary "W"

# Daily summary
crispt --inp corpus_folder --temporal-summary "D"
```

### 4. Temporal Sentiment Analysis

Track sentiment trends over time:

```python
# Weekly average sentiment
trend_df = analyzer.get_temporal_sentiment_trend(period="W", aggregation="mean")

# Daily median sentiment
trend_df = analyzer.get_temporal_sentiment_trend(period="D", aggregation="median")
```

**CLI Usage:**
```bash
# Weekly mean sentiment
crispt --inp corpus_folder --temporal-sentiment "W:mean"

# Daily median sentiment
crispt --inp corpus_folder --temporal-sentiment "D:median"
```

### 5. Temporal Topic Modeling

Extract topics over time periods:

```python
# Weekly top 5 topics
topics_dict = analyzer.get_temporal_topics(period="W", top_n=5)

# Daily top 10 topics
topics_dict = analyzer.get_temporal_topics(period="D", top_n=10)

# Result format: {"2025-W03": ["topic1", "topic2", ...], ...}
```

**CLI Usage:**
```bash
# Weekly top 5 topics
crispt --inp corpus_folder --temporal-topics "W:5"
```

## Temporal Visualization

### Sentiment Trends

```python
# Plot sentiment trend
fig = analyzer.plot_temporal_sentiment(
    period="W",
    aggregation="mean",
    output_path="sentiment_trend.png"
)
```

### Temporal Summary Plots

```python
# Plot temporal summary
fig = analyzer.plot_temporal_summary(
    time_column="timestamp",
    period="W",
    output_path="temporal_summary.png"
)
```

## Temporal Graph Analysis

### Creating Temporal Subgraphs

Generate time-sliced subgraphs:

```python
from crisp_t import CrispGraph

# Create base graph
graph_gen = CrispGraph(corpus)
graph_gen.create_graph()

# Create weekly subgraphs
subgraphs = graph_gen.create_temporal_subgraphs(period="W")

# Access specific period
week_graph = subgraphs["2025-W03"]
print(f"Nodes: {week_graph.number_of_nodes()}")
print(f"Edges: {week_graph.number_of_edges()}")
```

**CLI Usage:**
```bash
crispt --inp corpus_folder --temporal-subgraphs "W"
```

### Adding Temporal Edges

Add temporal relationship edges to existing graph:

```python
graph_gen = CrispGraph(corpus)
graph_gen.create_graph()

# Add temporal edges based on temporal_links in document metadata
num_edges = graph_gen.add_temporal_edges()
print(f"Added {num_edges} temporal edges")
```

## MCP Server Integration

The MCP server provides temporal analysis tools for AI assistants:

### Available Tools

1. **temporal_link_by_time** - Link documents to dataframe by timestamps
2. **temporal_filter** - Filter corpus by time range
3. **temporal_summary** - Generate temporal summary
4. **temporal_sentiment_trend** - Analyze sentiment trends
5. **temporal_topics** - Extract topics over time

### Example MCP Usage

```json
{
  "tool": "temporal_link_by_time",
  "arguments": {
    "method": "window",
    "time_column": "timestamp",
    "window_seconds": 300
  }
}
```

## Complete Workflow Example

Here's a complete workflow for temporal analysis:

### Step 1: Load Data with Timestamps

```python
from crisp_t import ReadData

# Load CSV with timestamps
read_data = ReadData()
read_data.read_csv_to_corpus(
    "data.csv",
    comma_separated_text_columns="notes,comments",
    timestamp_column="timestamp"  # Auto-detected if named conventionally
)
corpus = read_data.create_corpus(
    name="Healthcare Data",
    description="Patient notes with sensor readings"
)
```

### Step 2: Perform Temporal Linking

```python
from crisp_t import TemporalAnalyzer

analyzer = TemporalAnalyzer(corpus)

# Link documents to nearest sensor readings (within 5 minutes)
from datetime import timedelta
corpus = analyzer.link_by_time_window(
    time_column="timestamp",
    window_before=timedelta(minutes=5),
    window_after=timedelta(minutes=5)
)
```

### Step 3: Analyze Sentiment Over Time

```python
from crisp_t import Sentiment

# First, run sentiment analysis
sentiment_analyzer = Sentiment(corpus)
sentiment_analyzer.get_sentiment(documents=True)

# Then analyze temporal trends
trend_df = analyzer.get_temporal_sentiment_trend(period="W", aggregation="mean")
print(trend_df)

# Visualize
fig = analyzer.plot_temporal_sentiment(period="W", output_path="sentiment_trend.png")
```

### Step 4: Extract Temporal Topics

```python
from crisp_t import Text

# Run topic modeling
text_analyzer = Text(corpus)
corpus = text_analyzer.topics()

# Extract temporal topics
topics = analyzer.get_temporal_topics(period="W", top_n=5)

for period, topic_list in topics.items():
    print(f"{period}: {', '.join(topic_list)}")
```

### Step 5: Create Temporal Graph

```python
from crisp_t import CrispGraph

# Assign keywords first
corpus = text_analyzer.assign()

# Create temporal subgraphs
graph_gen = CrispGraph(corpus)
graph_gen.create_graph()
subgraphs = graph_gen.create_temporal_subgraphs(period="W")

# Analyze each period
for period, graph in subgraphs.items():
    print(f"\n{period}:")
    print(f"  Nodes: {graph.number_of_nodes()}")
    print(f"  Edges: {graph.number_of_edges()}")
```

### Step 6: Save Results

```python
# Save corpus with temporal analysis
read_data = ReadData(corpus=corpus)
read_data.write_corpus_to_json("output_folder", corpus=corpus)
```

## CLI Workflow Example

```bash
# 1. Load data and perform temporal linking
crispt --source data_folder \
       --temporal-link "window:timestamp:300" \
       --out temp_corpus

# 2. Run sentiment analysis
crisp --inp temp_corpus \
      --sentiment documents \
      --out temp_corpus

# 3. Analyze temporal sentiment
crispt --inp temp_corpus \
       --temporal-sentiment "W:mean" \
       --out temp_corpus

# 4. Run topic modeling
crisp --inp temp_corpus \
      --topics 8 \
      --assign \
      --out temp_corpus

# 5. Extract temporal topics
crispt --inp temp_corpus \
       --temporal-topics "W:5" \
       --out temp_corpus

# 6. Create temporal subgraphs
crispt --inp temp_corpus \
       --temporal-subgraphs "W" \
       --out final_corpus

# 7. View results
crispt --inp final_corpus --print
```

## Period Codes Reference

- `D` - Day
- `W` - Week (Monday to Sunday)
- `M` - Month
- `Y` - Year
- `H` - Hour (for high-frequency data)

## Best Practices

1. **Timestamp Format**: Use ISO 8601 format for best compatibility
2. **Time Zones**: Be consistent with time zones across your data
3. **Missing Timestamps**: Documents without timestamps are preserved but excluded from temporal analysis
4. **Performance**: For large datasets, use sequence-based linking rather than window-based
5. **Validation**: Always check temporal links before proceeding with analysis
6. **Granularity**: Choose appropriate time periods (D/W/M) based on data density

## Troubleshooting

### No temporal data available

**Problem**: Temporal functions return empty results

**Solutions**:
- Verify documents have `timestamp` field set
- Check CSV has timestamp column (auto-detected names: timestamp, datetime, time, date)
- Ensure timestamp format is parseable (ISO 8601 recommended)
- Use `corpus.documents[0].timestamp` to check first document

### Time parsing errors

**Problem**: Timestamps not recognized

**Solutions**:
- Convert to ISO 8601: `YYYY-MM-DDTHH:MM:SS`
- Use pandas to preprocess: `pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%dT%H:%M:%S')`
- Check for timezone information (include or exclude consistently)

### Links not created

**Problem**: Temporal linking produces no links

**Solutions**:
- Verify both documents AND dataframe have timestamps
- Check time ranges overlap between documents and dataframe
- For window method, try larger window (e.g., 600 seconds instead of 300)
- Use nearest method if window method fails

## Advanced Topics

### Custom Timestamp Columns

```python
# Specify custom timestamp column
analyzer.link_by_nearest_time(time_column="created_date")
```

### Multiple Time Periods

```python
# Analyze at different granularities
daily_summary = analyzer.get_temporal_summary(period="D")
weekly_summary = analyzer.get_temporal_summary(period="W")
monthly_summary = analyzer.get_temporal_summary(period="M")
```

### Temporal Relationships

```python
# Add explicit temporal relationship
analyzer.add_temporal_relationship(
    doc_id="doc1",
    df_column="sensor_value",
    relation="temporal_correlation"
)

# Retrieve temporal relationships
relationships = corpus.get_relationships()
temporal_rels = [r for r in relationships if "temporal" in r["relation"]]
```

## References

- [Pandas Period Documentation](https://pandas.pydata.org/docs/user_guide/timeseries.html#time-span-representation)
- [ISO 8601 Standard](https://en.wikipedia.org/wiki/ISO_8601)
- [Temporal Data Analysis in Mixed Methods Research](https://journals.sagepub.com/doi/10.1177/1558689808326765)
