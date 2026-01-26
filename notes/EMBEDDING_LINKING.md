# Embedding-based Cross-Modal Linking Guide

## Overview

Embedding-based cross-modal linking provides **fuzzy semantic alignment** between text documents and numeric data rows when explicit IDs or timestamps are missing. This complements CRISP-T's existing linking methods:

- **ID-based linking**: Exact matches using document IDs
- **Keyword-based linking**: Relationships through shared terms
- **Time-based linking**: Temporal alignment using timestamps
- **Embedding-based linking**: Semantic similarity in vector space (NEW)

## How It Works

### Text Embeddings
- Documents are encoded as dense vectors using sentence transformers
- Default model: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- Captures semantic meaning beyond keyword matches
- Uses ChromaDB for efficient storage and retrieval

### Numeric Embeddings
- Dataframe rows are encoded as standardized numeric vectors
- Each row becomes a point in multi-dimensional feature space
- Normalization ensures fair comparison across features
- Handles missing values automatically

### Similarity Metrics
- **Cosine Similarity**: Measures angle between vectors (recommended)
  - Range: -1 to 1 (higher = more similar)
  - Good for semantic matching
  
- **Euclidean Distance**: Measures geometric distance
  - Converted to similarity: `1 / (1 + distance)`
  - Good for numeric proximity

### Linking Strategy
- Nearest neighbor search in combined embedding space
- Optional threshold filtering for precision control
- Top-k selection for multiple candidates per document
- Links stored in document metadata for later retrieval

## Installation

Embedding-based linking requires ChromaDB:

```bash
pip install chromadb
```

## Python API

### Basic Usage

```python
from crisp_t.embedding_linker import EmbeddingLinker
from crisp_t.model import Corpus

# Assuming you have a corpus with documents and dataframe
linker = EmbeddingLinker(
    corpus,
    similarity_metric="cosine",
    use_simple_embeddings=False  # Use transformer embeddings
)

# Link documents to nearest dataframe row
corpus = linker.link_by_embedding_similarity(top_k=1)

# Check results
stats = linker.get_link_statistics()
print(f"Linked {stats['linked_documents']} documents")
print(f"Average similarity: {stats['avg_similarity']:.3f}")
```

### Advanced Options

```python
# Link to top 3 most similar rows with threshold
corpus = linker.link_by_embedding_similarity(
    top_k=3,                        # Top 3 candidates
    threshold=0.7,                  # Min similarity 0.7
    numeric_columns=["age", "bmi"]  # Use specific columns
)

# Visualize embedding space
fig = linker.visualize_embedding_space(
    output_path="embedding_viz.png",
    method="tsne"  # or "pca", "umap"
)
```

### Accessing Links

```python
# Links are stored in document metadata
for doc in corpus.documents:
    if "embedding_links" in doc.metadata:
        for link in doc.metadata["embedding_links"]:
            df_index = link["df_index"]
            similarity = link["similarity_score"]
            print(f"Document '{doc.name}' → Row {df_index} (sim: {similarity:.3f})")
            
            # Access linked row data
            row_data = corpus.df.iloc[df_index]
            print(f"  Temperature: {row_data['temperature']}")
```

## CLI Usage

### Basic Linking

```bash
# Link using cosine similarity, top-1 match
crispt --inp corpus_folder --embedding-link "cosine:1" --out corpus_folder

# Link with threshold (only similarities >= 0.7)
crispt --inp corpus_folder --embedding-link "cosine:1:0.7" --out corpus_folder

# Link with multiple candidates per document
crispt --inp corpus_folder --embedding-link "cosine:3:0.6" --out corpus_folder

# Use Euclidean distance instead
crispt --inp corpus_folder --embedding-link "euclidean:1" --out corpus_folder
```

### View Statistics

```bash
# Display embedding link statistics
crispt --inp corpus_folder --embedding-stats
```

### Visualization

```bash
# Visualize with t-SNE
crispt --inp corpus_folder --embedding-viz "tsne:embedding_space.png"

# Visualize with PCA
crispt --inp corpus_folder --embedding-viz "pca:embedding_pca.png"
```

## MCP Server Integration

The MCP server provides two tools for embedding-based linking:

### embedding_link

Link documents to dataframe rows using embedding similarity.

```json
{
  "tool": "embedding_link",
  "arguments": {
    "similarity_metric": "cosine",
    "top_k": 1,
    "threshold": 0.7,
    "numeric_columns": "temperature,heart_rate"
  }
}
```

### embedding_link_stats

Get statistics about existing embedding links.

```json
{
  "tool": "embedding_link_stats",
  "arguments": {}
}
```

## Use Cases

### Healthcare: Patient Notes ↔ Vital Signs

Link free-text clinical notes to vital sign measurements when timestamps are imprecise:

```python
# Notes: "Patient feeling tired with headache"
# Vitals: temperature=38.2, heart_rate=95, bp_sys=125
# → High similarity due to fever-related symptoms

linker = EmbeddingLinker(corpus, similarity_metric="cosine")
corpus = linker.link_by_embedding_similarity(
    numeric_columns=["temperature", "heart_rate", "bp_sys"],
    top_k=2,
    threshold=0.6
)
```

### Social Media: Posts ↔ Engagement Metrics

Link social media posts to engagement data when post IDs are unavailable:

```python
# Post: "Amazing new product launch!"
# Metrics: likes=500, shares=120, sentiment_score=0.85
# → High similarity for positive sentiment + high engagement

linker = EmbeddingLinker(corpus, similarity_metric="cosine")
corpus = linker.link_by_embedding_similarity(
    numeric_columns=["likes", "shares", "sentiment_score"],
    top_k=1
)
```

### Education: Feedback ↔ Performance

Link student feedback to academic performance metrics:

```python
# Feedback: "Excellent teaching, very clear explanations"
# Performance: exam_score=92, attendance=0.95, participation=4.5
# → High similarity for positive feedback + strong performance

linker = EmbeddingLinker(corpus)
corpus = linker.link_by_embedding_similarity(
    numeric_columns=["exam_score", "attendance", "participation"],
    threshold=0.65
)
```

## Combining with Other Linking Methods

Embedding-based linking works alongside existing methods:

```python
from crisp_t.temporal import TemporalAnalyzer
from crisp_t.embedding_linker import EmbeddingLinker

# Step 1: Time-based linking (if timestamps available)
temp_analyzer = TemporalAnalyzer(corpus)
corpus = temp_analyzer.link_by_time_window(
    time_column="timestamp",
    window_before=timedelta(minutes=5),
    window_after=timedelta(minutes=5)
)

# Step 2: Embedding-based linking for additional fuzzy matches
emb_linker = EmbeddingLinker(corpus)
corpus = emb_linker.link_by_embedding_similarity(
    threshold=0.7,
    top_k=2
)

# Documents now have both temporal_links and embedding_links
for doc in corpus.documents:
    print(f"\nDocument: {doc.name}")
    if "temporal_links" in doc.metadata:
        print(f"  Temporal links: {len(doc.metadata['temporal_links'])}")
    if "embedding_links" in doc.metadata:
        print(f"  Embedding links: {len(doc.metadata['embedding_links'])}")
```

## Performance Considerations

### Embedding Generation
- **First-time cost**: Generating embeddings can be slow for large corpora
- **Caching**: Embeddings are cached within a session
- **Batch processing**: Documents processed in batches for efficiency

### Similarity Computation
- **Memory**: O(n_docs × n_rows × embedding_dim)
- **Time**: O(n_docs × n_rows) for similarity calculation
- **Optimization**: Use specific numeric columns to reduce dimensionality

### Recommendations
- For corpora >10,000 documents: Use simple embeddings or batch processing
- For dataframes >100,000 rows: Pre-filter rows before linking
- For repeated analyses: Save corpus with links to avoid recomputation

## Threshold Selection

Choosing the right similarity threshold is important for precision:

- **threshold=0.9**: Very strict, only near-perfect semantic matches
- **threshold=0.7**: Balanced, good semantic alignment
- **threshold=0.5**: Permissive, allows loose associations
- **threshold=None**: No filtering, keep all top-k matches

Example threshold analysis:

```python
# Try different thresholds
for threshold in [0.5, 0.7, 0.9]:
    linker = EmbeddingLinker(corpus)
    test_corpus = linker.link_by_embedding_similarity(
        threshold=threshold,
        top_k=1
    )
    stats = linker.get_link_statistics()
    print(f"Threshold {threshold}: {stats['linked_documents']} linked")
```

## Troubleshooting

### No links created

**Problem**: `linked_documents = 0`

**Solutions**:
- Lower the threshold (try 0.5 instead of 0.7)
- Increase top_k to get more candidates
- Check that dataframe has numeric columns
- Verify documents have meaningful text content

### Low similarity scores

**Problem**: Average similarity < 0.5

**Solutions**:
- Check if text and numeric data are truly related
- Use specific numeric columns that relate to text content
- Consider combining with temporal or ID-based linking
- Review data quality (missing values, outliers)

### ChromaDB errors

**Problem**: ImportError or ChromaDB failures

**Solutions**:
- Install ChromaDB: `pip install chromadb`
- Use simple embeddings: `use_simple_embeddings=True`
- Check ChromaDB version compatibility
- Clear ChromaDB cache if corruption suspected

### Dimension mismatch

**Problem**: Text and numeric embeddings have different dimensions

**Solutions**:
- Don't worry! The linker automatically pads shorter embeddings
- This may reduce accuracy slightly
- Consider using PCA to reduce numeric dimensionality first

## Advanced Topics

### Custom Embedding Models

```python
# Use different sentence transformer model
linker = EmbeddingLinker(
    corpus,
    text_embedding_model="all-mpnet-base-v2",  # Larger, more accurate
    similarity_metric="cosine"
)
```

### Dimensionality Reduction

```python
from sklearn.decomposition import PCA

# Reduce numeric embedding dimensions
pca = PCA(n_components=50)
reduced_embeddings = pca.fit_transform(linker._get_numeric_embeddings())
```

### Hybrid Scoring

```python
# Combine embedding similarity with temporal proximity
def hybrid_score(emb_sim, time_gap, alpha=0.7):
    time_sim = 1.0 / (1.0 + abs(time_gap))
    return alpha * emb_sim + (1 - alpha) * time_sim

# Apply custom scoring to links
for doc in corpus.documents:
    if "embedding_links" in doc.metadata and "temporal_links" in doc.metadata:
        # Custom logic to combine both link types
        pass
```

## References

- [Sentence Transformers](https://www.sbert.net/) - Text embedding models
- [ChromaDB Documentation](https://docs.trychroma.com/) - Vector database
- [Cross-modal retrieval research](https://arxiv.org/abs/2104.07081)
- [CRISP-T Triangulation Guide](./INSTRUCTION.md)

## Best Practices

1. **Start with defaults**: cosine similarity, top_k=1
2. **Experiment with thresholds**: Find balance between recall and precision
3. **Visualize embeddings**: Use t-SNE to verify meaningful structure
4. **Validate links**: Manually check a sample of links for semantic correctness
5. **Combine methods**: Use with temporal/ID linking for robust triangulation
6. **Document choices**: Record threshold and metric selections in metadata
7. **Save frequently**: Embedding computation is expensive, save results

## Example Workflow

Complete workflow demonstrating embedding-based linking:

```python
from crisp_t.read_data import ReadData
from crisp_t.embedding_linker import EmbeddingLinker
from crisp_t.sentiment import Sentiment

# 1. Load data
read_data = ReadData()
read_data.read_csv_to_corpus(
    "data.csv",
    comma_separated_text_columns="notes,comments"
)
corpus = read_data.create_corpus(name="Research Data")

# 2. Run sentiment analysis (optional, adds context)
sentiment = Sentiment(corpus)
sentiment.get_sentiment(documents=True)

# 3. Embedding-based linking
linker = EmbeddingLinker(corpus, similarity_metric="cosine")
corpus = linker.link_by_embedding_similarity(
    threshold=0.7,
    top_k=2,
    numeric_columns=["score", "rating", "engagement"]
)

# 4. Review statistics
stats = linker.get_link_statistics()
print(f"Linked: {stats['linked_documents']}/{stats['total_documents']}")
print(f"Avg similarity: {stats['avg_similarity']:.3f}")

# 5. Visualize
fig = linker.visualize_embedding_space(output_path="viz.png", method="tsne")

# 6. Save results
read_data.write_corpus_to_json("output_folder", corpus=corpus)
```
