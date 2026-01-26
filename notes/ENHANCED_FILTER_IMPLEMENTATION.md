# Enhanced Filter with Linkage Support - Implementation Notes

## Overview

The `--filter` CLI option has been extended to support multiple linkage methods, enabling sophisticated filtering based on different types of relationships between text documents and numeric data.

## Previous Behavior (Legacy)

**Format**: `--filter key=value` or `--filter key:value`

**Functionality**: 
- Filtered dataframe rows where `column == value`
- Filtered documents by matching IDs (if ID column present)
- Only supported ID-based filtering

## New Behavior (Enhanced)

**Formats**:
1. **Legacy format** (backward compatible): `key=value`
   - Defaults to ID-based filtering
   - Example: `--filter Gender=1`

2. **Method-based format**: `method:key=value`
   - Specify linkage method explicitly
   - Example: `--filter id:Gender=1`

3. **With threshold**: `method:key=value:threshold`
   - For embedding-based filtering with similarity threshold
   - Example: `--filter embedding:similarity:0.7`

4. **Method with key only**: `method:key`
   - For filters that don't require a value
   - Example: `--filter time:timestamp` or `--filter keyword:healthcare`

## Supported Linkage Methods

### 1. ID-based Filtering (`id`)

**Usage**: `--filter id:column=value`

**Behavior**:
- Filters dataframe rows where `column == value`
- Filters documents with matching IDs from filtered rows
- Requires ID column in dataframe

**Example**:
```bash
crisp --inp corpus --filter id:Gender=1 --out filtered
```

**Error Messages**:
- "No dataframe available for ID-based filtering"
- "Column 'X' not found in dataframe"

### 2. Keyword-based Filtering (`keyword`)

**Usage**: `--filter keyword:keyword_name`

**Behavior**:
- Filters documents that have the keyword in metadata
- Uses relationships to filter linked dataframe columns
- Requires keywords assigned (via topic modeling)

**Example**:
```bash
# First assign keywords
crisp --inp corpus --topics --assign --out corpus

# Then filter by keyword
crisp --inp corpus --filter keyword:healthcare --out filtered
```

**Error Messages**:
- "No documents found with keyword 'X'. Hint: Run topic modeling..."

### 3. Time-based Filtering (`time`)

**Usage**: `--filter time:column_name`

**Behavior**:
- Filters documents that have temporal links
- Filters dataframe rows linked via temporal metadata
- Requires temporal_links in document metadata

**Example**:
```bash
# First create temporal links
crispt --inp corpus --temporal-link "nearest:timestamp" --out corpus

# Then filter by time
crisp --inp corpus --filter time:timestamp --out filtered
```

**Error Messages**:
- "No temporal links found in corpus. Hint: Create temporal links first..."
- Instructions on how to create temporal links

### 4. Embedding-based Filtering (`embedding`)

**Usage**: `--filter embedding:key:threshold`

**Behavior**:
- Filters documents with embedding links above similarity threshold
- Filters dataframe rows linked via embedding metadata
- Requires embedding_links in document metadata

**Example**:
```bash
# First create embedding links
crispt --inp corpus --embedding-link "cosine:1:0.7" --out corpus

# Then filter by embedding similarity >= 0.8
crisp --inp corpus --filter embedding:similarity:0.8 --out filtered
```

**Error Messages**:
- "No embedding links found in corpus. Hint: Create embedding links first..."
- "No embedding links with similarity >= X"

## Implementation Details

### Filter Parsing Logic

```python
def parse_filter(filter_string):
    """
    Parse filter string into components.
    
    Examples:
      'Gender=1' → method='id', key='Gender', value='1', threshold=None
      'id:Gender=1' → method='id', key='Gender', value='1', threshold=None
      'keyword:healthcare' → method='keyword', key='healthcare', value='', threshold=None
      'time:timestamp' → method='time', key='timestamp', value='', threshold=None
      'embedding:sim:0.7' → method='embedding', key='sim', value='', threshold=0.7
    """
```

### Filter Application Sequence

1. **Parse filter format** - Determine method, key, value, threshold
2. **Validate linkage metadata** - Check if required links exist
3. **Apply filter** - Use LinkageFilter class for non-ID methods
4. **Update corpus** - Return filtered documents and dataframe
5. **Report results** - Show count of filtered documents/rows

### LinkageFilter Class

**Location**: `src/crisp_t/linkage_filter.py`

**Key Methods**:
- `filter_by_linkage(method, key, value, min_similarity)` - Main filter entry point
- `_filter_by_id(column, value)` - ID-based filtering
- `_filter_by_keyword(keyword, value)` - Keyword-based filtering
- `_filter_by_time(time_column, value)` - Time-based filtering
- `_filter_by_embedding(key, value, min_similarity)` - Embedding-based filtering
- `get_filter_statistics(method)` - Get linkage availability stats

**Error Handling**:
- Validates linkage method
- Checks for required metadata
- Provides helpful error messages with hints
- Suggests commands to create missing linkage

## Backward Compatibility

**Legacy format preserved**: `--filter key=value`
- Automatically treated as ID-based filtering
- No breaking changes to existing scripts
- Works exactly as before

**Migration path**:
- Old: `--filter Gender=1`
- New (equivalent): `--filter id:Gender=1`
- Both work identically

## Usage Examples

### Example 1: Healthcare Data Analysis

```bash
# Load data
crisp --source healthcare_data --out corpus

# Create temporal links (patient notes to vital signs)
crispt --inp corpus --temporal-link "window:timestamp:300" --out corpus

# Filter to documents with temporal links
crisp --inp corpus --filter time:timestamp --out time_filtered

# Run sentiment analysis on filtered data
crisp --inp time_filtered --sentiment --out time_filtered
```

### Example 2: Multi-stage Filtering

```bash
# Create keyword and embedding links
crisp --inp corpus --topics --assign --out corpus
crispt --inp corpus --embedding-link "cosine:2:0.6" --out corpus

# Filter by keyword first
crisp --inp corpus --filter keyword:healthcare --out stage1

# Then filter by high embedding similarity
crisp --inp stage1 --filter embedding:sim:0.8 --out stage2
```

### Example 3: Combined with Analysis

```bash
# Filter by gender, then analyze
crisp --inp corpus --filter id:Gender=1 --sentiment --topics --out female_analysis

# Filter by temporal links, then cluster
crispt --inp corpus --temporal-link "sequence:timestamp:W" --out corpus
crisp --inp corpus --filter time:timestamp --kmeans --out temporal_clusters
```

## Testing

**Test Coverage**:
- 20 unit tests for LinkageFilter class
- Tests for each linkage method (id, keyword, time, embedding)
- Error handling tests (missing metadata, invalid methods)
- Sequential filtering tests
- Edge cases (empty results, missing dataframe)

**Test File**: `tests/test_linkage_filter.py`

**Run Tests**:
```bash
pytest tests/test_linkage_filter.py -v
```

## Performance Considerations

### Memory Usage
- Filtering creates new Corpus objects (doesn't modify in-place)
- Minimal memory overhead for metadata copying
- Efficient for datasets < 100K documents

### Time Complexity
- ID filtering: O(n) where n = dataframe rows
- Keyword filtering: O(d) where d = documents
- Time filtering: O(d × links_per_doc)
- Embedding filtering: O(d × links_per_doc)

### Optimization Tips
- Apply most restrictive filters first
- Use ID filtering when possible (fastest)
- Cache intermediate filtered corpora
- Combine multiple filters in single command when appropriate

## Error Messages and Hints

All error messages include:
1. **What went wrong** - Clear description of the error
2. **Why it happened** - Missing metadata or invalid input
3. **How to fix it** - Specific command to create required linkage

**Example Error**:
```
❌ Filter Error: No temporal links found in corpus.

Hint: Create temporal links first:
  crispt --inp <folder> --temporal-link 'nearest:timestamp' --out <folder>
  or use window/sequence methods. See notes/TEMPORAL_ANALYSIS.md
```

## Integration with Other Features

### Works with all analysis commands
```bash
# Filter + Text Analysis
crisp --inp corpus --filter keyword:healthcare --topics --sentiment

# Filter + ML Analysis
crisp --inp corpus --filter id:Gender=1 --kmeans --pca

# Filter + Graph Generation
crispt --inp corpus --filter time:timestamp --graph
```

### Chainable with temporal/embedding linking
```bash
# Create links, filter, analyze in one workflow
crispt --inp corpus \
       --temporal-link "window:timestamp:300" \
       --embedding-link "cosine:1:0.7" \
       --out corpus

crisp --inp corpus --filter embedding:sim:0.8 --sentiment --out results
```

## Future Enhancements

Potential additions (not in current scope):
- Range-based filtering (e.g., `similarity:0.7-0.9`)
- Negation filters (e.g., `!keyword:healthcare`)
- Regex pattern matching
- Custom filter functions
- Filter composition (AND/OR logic)
- Filter preview mode (show what would be filtered)

## Migration Guide

For users upgrading from older versions:

**No changes required** - Legacy `--filter key=value` continues to work

**Optional enhancements**:
1. Use explicit method: `--filter id:key=value` (clearer intent)
2. Explore new methods: keyword, time, embedding
3. Create linkage metadata for advanced filtering
4. Combine filters for sophisticated analysis

## Conclusion

The enhanced `--filter` functionality provides:
- **Backward compatibility** - No breaking changes
- **Multiple linkage methods** - Flexible filtering options
- **Helpful error messages** - Clear guidance for users
- **Comprehensive testing** - 20 unit tests
- **Full documentation** - Usage examples and best practices

This enables researchers to perform sophisticated filtering based on semantic, temporal, and similarity-based relationships in their mixed-methods data.
