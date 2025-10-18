# Semantic Search Examples

This document demonstrates using the semantic search features in CRISP-T.

## Command Line Examples

### Example 1: Basic Semantic Search

Search for documents similar to a query:

```bash
crispt \
  --id medical_corpus \
  --name "Medical Research" \
  --doc "1|AI in Healthcare|Machine learning and artificial intelligence applications in medical diagnosis" \
  --doc "2|NLP Research|Natural language processing techniques for clinical text analysis" \
  --doc "3|Patient Care|Improving patient outcomes through evidence-based medicine" \
  --doc "4|Drug Discovery|Using computational methods for pharmaceutical research" \
  --semantic "artificial intelligence medical" \
  --num 2 \
  --print
```

This will find the 2 most similar documents to the query "artificial intelligence medical".

### Example 2: Find Similar Documents (Literature Review)

Find documents similar to a reference document (useful for literature reviews):

```bash
crispt \
  --id research_corpus \
  --name "Research Papers" \
  --doc "1|Paper A|Deep learning for medical image analysis" \
  --doc "2|Paper B|Transfer learning in computer vision" \
  --doc "3|Paper C|Natural language understanding models" \
  --doc "4|Paper D|Convolutional neural networks for diagnosis" \
  --similar-docs "1" \
  --num 2 \
  --rec 0.7 \
  --print
```

This finds 2 documents most similar to document "1", with a similarity threshold of 0.7.

You can also search based on multiple reference documents:

```bash
crispt \
  --inp ./corpus_folder \
  --similar-docs "1,2,5" \
  --num 5 \
  --rec 0.6
```

This finds documents similar to documents 1, 2, and 5 combined, returning up to 5 results with similarity above 0.6.

### Example 3: Export Metadata to DataFrame

Create a corpus and export document metadata as a DataFrame:

```bash
crispt \
  --id research_papers \
  --name "Research Papers" \
  --doc "1|Paper A|Deep learning for image classification" \
  --doc "2|Paper B|Transfer learning in computer vision" \
  --doc "3|Paper C|Natural language understanding models" \
  --metadata-df \
  --df-cols \
  --out ./output
```

This creates a DataFrame with document IDs and metadata, which can be saved for further analysis.

### Example 3: Filter and Export

Combine semantic search with DataFrame export:

```bash
crispt \
  --inp ./corpus_folder \
  --semantic "machine learning healthcare" \
  --num 10 \
  --metadata-df \
  --metadata-keys "topic,year,author" \
  --out ./filtered_corpus
```

## Python API Examples

### Example 1: Basic Semantic Search

```python
from crisp_t.model import Corpus, Document
from crisp_t.semantic import Semantic

# Create corpus
docs = [
    Document(
        id='1',
        name='AI Doc',
        text='Machine learning and artificial intelligence',
        metadata={'topic': 'tech', 'year': '2024'}
    ),
    Document(
        id='2',
        name='NLP Doc',
        text='Natural language processing',
        metadata={'topic': 'nlp', 'year': '2023'}
    ),
]
corpus = Corpus(id='test', name='Test', description='Test', documents=docs)

# Perform semantic search
semantic = Semantic(corpus)
result = semantic.get_similar('artificial intelligence', n_results=1)

print(f"Found {len(result.documents)} documents")
for doc in result.documents:
    print(f"- {doc.name}: {doc.text[:50]}...")
```

### Example 2: Find Similar Documents

```python
from crisp_t.model import Corpus, Document
from crisp_t.semantic import Semantic

# Create corpus
docs = [
    Document(id='1', text='Machine learning and AI'),
    Document(id='2', text='Natural language processing'),
    Document(id='3', text='Healthcare data analysis'),
    Document(id='4', text='Deep learning neural networks'),
]
corpus = Corpus(id='research', documents=docs)

# Find documents similar to document 1
semantic = Semantic(corpus)
similar_ids = semantic.get_similar_documents(
    document_ids='1',
    n_results=2,
    threshold=0.7
)

print(f"Similar documents: {similar_ids}")
# Output: Similar documents: ['4', '2']

# Find documents similar to multiple reference docs
similar_ids = semantic.get_similar_documents(
    document_ids='1,2',
    n_results=2,
    threshold=0.6
)
print(f"Similar to 1 and 2: {similar_ids}")
```

### Example 3: Export Metadata

```python
from crisp_t.model import Corpus, Document
from crisp_t.semantic import Semantic

# Create corpus with metadata
docs = [
    Document(
        id='doc1',
        text='Healthcare AI research',
        metadata={'category': 'medical', 'score': 0.95}
    ),
    Document(
        id='doc2',
        text='Computer vision applications',
        metadata={'category': 'vision', 'score': 0.87}
    ),
]
corpus = Corpus(id='research', documents=docs)

# Export metadata to DataFrame
semantic = Semantic(corpus)
result = semantic.get_df(metadata_keys=['category', 'score'])

print(result.df)
# Output:
#      id category  score
# 0  doc1  medical   0.95
# 1  doc2   vision   0.87
```

### Example 3: Save and Restore Collection

```python
from crisp_t.semantic import Semantic

# Create and save
semantic = Semantic(corpus)
semantic.save_collection('./my_collection')

# Later, restore
new_semantic = Semantic(new_corpus)
new_semantic.restore_collection('./my_collection')
result = new_semantic.get_similar('query text', n_results=5)
```

## MCP Tool Examples

When using CRISP-T's MCP server with an AI assistant:

### Semantic Search

```
Use the semantic_search tool with:
- query: "machine learning healthcare"
- n_results: 5

This will find the 5 most similar documents and update the current corpus.
```

### Find Similar Documents (Literature Review)

```
Use the find_similar_documents tool with:
- document_ids: "doc1,doc2"
- n_results: 5
- threshold: 0.7

This will find documents similar to doc1 and doc2, returning up to 5 results 
with similarity above 0.7. This is particularly useful for literature reviews 
where you want to find additional relevant papers similar to known good examples.
```

### Export Metadata

```
Use the export_metadata_df tool with:
- metadata_keys: "topic,year,author"

This will create a DataFrame with the specified metadata fields.
```

## Notes

- By default, ChromaDB uses ONNX MiniLM-L6-V2 embeddings which require a download on first use
- If network is unavailable, the system will automatically fall back to simple bag-of-words embeddings
- For production use, consider pre-downloading the embedding model or using a persistent ChromaDB instance
- Semantic search works best with longer, descriptive text documents
- The simple embeddings fallback is suitable for testing and offline use, but production should use the default embeddings for better quality

## Integration with Existing Workflows

Semantic search can be combined with other CRISP-T features:

1. **Topic Modeling + Semantic Search**: First use topic modeling to assign keywords, then use semantic search to find similar documents based on those topics
2. **Coding Dictionary + Metadata Export**: Extract coding dictionaries, then export metadata to analyze patterns across documents
3. **Triangulation**: Use semantic search to group similar qualitative responses, then correlate with quantitative data using relationships
