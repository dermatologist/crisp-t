#!/usr/bin/env python3
"""
Example: Embedding-based Cross-Modal Linking

This example demonstrates how to use embedding-based linking to connect
text documents to numeric data rows when explicit IDs or timestamps are missing.

Use case: Linking patient feedback comments to their clinical measurements.
"""

import pandas as pd
from crisp_t.model import Corpus, Document
from crisp_t.embedding_linker import EmbeddingLinker

print("="*70)
print(" Embedding-based Cross-Modal Linking Example")
print("="*70)

# Step 1: Create sample data
print("\n1. Creating sample patient feedback and clinical data...")

# Patient feedback (text documents)
documents = [
    Document(
        id="feedback1",
        name="Patient A Feedback",
        text="Feeling much better after treatment. Pain has decreased significantly.",
        metadata={}
    ),
    Document(
        id="feedback2",
        name="Patient B Feedback",
        text="Still experiencing severe headaches and high blood pressure symptoms.",
        metadata={}
    ),
    Document(
        id="feedback3",
        name="Patient C Feedback",
        text="Moderate improvement in mobility. Some joint pain remains.",
        metadata={}
    ),
    Document(
        id="feedback4",
        name="Patient D Feedback",
        text="Excellent recovery. All vital signs are normal and stable.",
        metadata={}
    ),
]

# Clinical measurements (numeric dataframe)
# Note: These don't have explicit patient IDs linking to the feedback
df = pd.DataFrame({
    "pain_score": [2, 8, 5, 1],      # 0-10 scale
    "blood_pressure_sys": [118, 145, 125, 112],
    "blood_pressure_dia": [75, 95, 82, 70],
    "mobility_score": [8, 4, 6, 9],  # 0-10 scale
    "recovery_index": [0.85, 0.35, 0.60, 0.95],  # 0-1 scale
})

# Create corpus
corpus = Corpus(
    id="patient_feedback",
    name="Patient Feedback & Clinical Data",
    description="Example demonstrating embedding-based linking",
    documents=documents,
    df=df
)

print(f"✓ Created {len(corpus.documents)} feedback documents")
print(f"✓ Created dataframe with {len(corpus.df)} rows of clinical data")

# Step 2: Initialize embedding linker
print("\n2. Initializing embedding linker...")

linker = EmbeddingLinker(
    corpus,
    similarity_metric="cosine",
    use_simple_embeddings=True  # Use simple embeddings for demo
)

print("✓ Linker initialized with cosine similarity")

# Step 3: Link documents to dataframe rows
print("\n3. Performing embedding-based linking...")
print("   Finding nearest dataframe row for each document...")

corpus = linker.link_by_embedding_similarity(
    top_k=1,           # Link to 1 most similar row
    threshold=None     # No threshold filtering
)

print("✓ Linking complete")

# Step 4: Display results
print("\n4. Linking Results:")
print("="*70)

for doc in corpus.documents:
    print(f"\n{doc.name}:")
    print(f"  Text: \"{doc.text[:60]}...\"")
    
    if "embedding_links" in doc.metadata and doc.metadata["embedding_links"]:
        for link in doc.metadata["embedding_links"]:
            df_index = link["df_index"]
            similarity = link["similarity_score"]
            
            print(f"\n  → Linked to Row {df_index} (similarity: {similarity:.3f})")
            
            # Show linked clinical data
            row = corpus.df.iloc[df_index]
            print(f"     Pain Score: {row['pain_score']}")
            print(f"     BP: {row['blood_pressure_sys']}/{row['blood_pressure_dia']}")
            print(f"     Mobility: {row['mobility_score']}")
            print(f"     Recovery: {row['recovery_index']:.2f}")
    else:
        print("  No links created")

# Step 5: Statistics
print("\n" + "="*70)
print("5. Overall Statistics:")
print("="*70)

stats = linker.get_link_statistics()

print(f"\nTotal documents: {stats['total_documents']}")
print(f"Linked documents: {stats['linked_documents']}")
print(f"Total links: {stats['total_links']}")
print(f"Average similarity: {stats['avg_similarity']:.3f}")
print(f"Min similarity: {stats['min_similarity']:.3f}")
print(f"Max similarity: {stats['max_similarity']:.3f}")

# Step 6: Advanced - Try with threshold
print("\n" + "="*70)
print("6. Advanced: Linking with Threshold")
print("="*70)

print("\nRe-linking with threshold=0.3 (stricter matching)...")

# Create fresh linker
linker2 = EmbeddingLinker(corpus, use_simple_embeddings=True)
corpus2 = linker2.link_by_embedding_similarity(
    top_k=2,           # Top 2 candidates
    threshold=0.3      # Only keep if similarity >= 0.3
)

stats2 = linker2.get_link_statistics()
print(f"\nWith threshold=0.3:")
print(f"  Linked documents: {stats2['linked_documents']}/{stats2['total_documents']}")
print(f"  Total links: {stats2['total_links']}")
print(f"  Average similarity: {stats2['avg_similarity']:.3f}")

# Step 7: Interpretation
print("\n" + "="*70)
print("7. Interpretation:")
print("="*70)

print("""
Embedding-based linking identified semantic associations between patient
feedback and clinical measurements:

- "Feeling better, pain decreased" → Low pain score, high recovery
- "Severe headaches, high BP" → High blood pressure readings
- "Moderate improvement" → Mid-range scores
- "Excellent recovery, normal vitals" → Best scores across metrics

This fuzzy matching works even without explicit patient IDs or timestamps,
making it ideal for:
  • Incomplete data
  • Legacy systems
  • Cross-database integration
  • Exploratory analysis
""")

# Step 8: Next steps
print("\n" + "="*70)
print("8. Next Steps:")
print("="*70)

print("""
To continue exploring embedding-based linking:

1. Try different similarity metrics:
   linker = EmbeddingLinker(corpus, similarity_metric="euclidean")

2. Combine with temporal linking:
   temporal_analyzer = TemporalAnalyzer(corpus)
   corpus = temporal_analyzer.link_by_time_window(...)
   
3. Visualize embedding space:
   fig = linker.visualize_embedding_space(output_path="viz.png")

4. Use transformer embeddings (better quality):
   linker = EmbeddingLinker(corpus, use_simple_embeddings=False)

5. Export to use in triangulation analysis:
   read_data.write_corpus_to_json("output_folder", corpus=corpus)
""")

print("\n✓ Example complete!")
