#!/usr/bin/env python3
"""
Example: Temporal Analysis of Patient Notes and Sensor Data

This example demonstrates how to:
1. Load text data (patient notes) and numeric data (sensor readings) with timestamps
2. Link them temporally
3. Analyze sentiment trends over time
4. Extract topics per time period
5. Visualize temporal patterns
"""

from datetime import timedelta

import pandas as pd

from crisp_t.model import Corpus, Document
from crisp_t.read_data import ReadData
from crisp_t.sentiment import Sentiment
from crisp_t.temporal import TemporalAnalyzer

# Step 1: Create sample data with timestamps
print("Creating sample temporal data...")

# Create sample patient notes with timestamps
documents = [
    Document(
        id="note1",
        name="Morning Assessment",
        text="Patient reports feeling much better today. Pain level decreased significantly.",
        timestamp="2025-01-15T09:00:00",
        metadata={}
    ),
    Document(
        id="note2",
        name="Afternoon Check",
        text="Patient experiencing some discomfort but overall stable condition.",
        timestamp="2025-01-15T14:30:00",
        metadata={}
    ),
    Document(
        id="note3",
        name="Evening Note",
        text="Patient resting well. No complaints. Vital signs normal.",
        timestamp="2025-01-15T20:00:00",
        metadata={}
    ),
    Document(
        id="note4",
        name="Next Day Morning",
        text="Patient shows continued improvement. Ready for discharge planning.",
        timestamp="2025-01-16T09:30:00",
        metadata={}
    ),
    Document(
        id="note5",
        name="Next Day Afternoon",
        text="Some anxiety about discharge but physically stable.",
        timestamp="2025-01-16T15:00:00",
        metadata={}
    ),
]

# Create sample sensor/numeric data
df = pd.DataFrame({
    'timestamp': [
        "2025-01-15T09:05:00",
        "2025-01-15T14:35:00",
        "2025-01-15T20:05:00",
        "2025-01-16T09:35:00",
        "2025-01-16T15:05:00",
    ],
    'heart_rate': [72, 78, 68, 70, 82],
    'blood_pressure_sys': [120, 125, 118, 122, 130],
    'blood_pressure_dia': [80, 82, 78, 80, 85],
    'pain_score': [6, 4, 2, 1, 2],
})

# Create corpus
corpus = Corpus(
    id="patient_temporal",
    name="Patient Temporal Data",
    description="Patient notes with sensor readings over time",
    documents=documents,
    df=df
)

print(f"✓ Created corpus with {len(corpus.documents)} documents and {len(corpus.df)} sensor readings")

# Step 2: Temporal Linking
print("\nPerforming temporal linking...")
analyzer = TemporalAnalyzer(corpus)

# Link documents to nearest sensor readings (within 10 minutes)
corpus = analyzer.link_by_time_window(
    time_column="timestamp",
    window_before=timedelta(minutes=10),
    window_after=timedelta(minutes=10)
)

# Check links
linked_count = sum(1 for doc in corpus.documents if "temporal_links" in doc.metadata)
print(f"✓ Linked {linked_count} documents to sensor readings")

# Show example link
for doc in corpus.documents[:2]:
    if "temporal_links" in doc.metadata:
        links = doc.metadata["temporal_links"]
        print(f"\nDocument '{doc.name}' linked to {len(links)} sensor reading(s):")
        for link in links:
            print(f"  - Row {link['df_index']}, time gap: {link['time_gap_seconds']:.1f}s")

# Step 3: Sentiment Analysis Over Time
print("\n" + "="*60)
print("SENTIMENT ANALYSIS OVER TIME")
print("="*60)

# Run sentiment analysis on documents
sentiment_analyzer = Sentiment(corpus)
sentiment_scores = sentiment_analyzer.get_sentiment(documents=True, verbose=False)

# Get temporal sentiment trend
trend_df = analyzer.get_temporal_sentiment_trend(period="D", aggregation="mean")
print("\nDaily Sentiment Trend:")
print(trend_df)

# Step 4: Temporal Summary
print("\n" + "="*60)
print("TEMPORAL SUMMARY")
print("="*60)

summary = analyzer.get_temporal_summary(
    time_column="timestamp",
    period="D",
    numeric_columns=["heart_rate", "pain_score"]
)

print("\nDaily Summary of Sensor Data:")
print(summary)

# Step 5: Topic Analysis Over Time
print("\n" + "="*60)
print("TOPIC ANALYSIS OVER TIME")
print("="*60)

# For temporal topic extraction, we'll use keyword extraction from text
# Note: For better results, run full topic modeling first with Text.topics()

# Get temporal topics
topics = analyzer.get_temporal_topics(period="D", top_n=3)

print("\nDaily Topics (keyword-based):")
for period, topic_list in topics.items():
    print(f"{period}: {', '.join(topic_list)}")

# Step 6: Visualize (optional - requires matplotlib)
print("\n" + "="*60)
print("VISUALIZATION")
print("="*60)

try:
    # Plot sentiment trend
    fig1 = analyzer.plot_temporal_sentiment(
        period="D",
        aggregation="mean",
        output_path="/tmp/sentiment_trend.png"
    )
    print("✓ Sentiment trend plot saved to /tmp/sentiment_trend.png")

    # Plot temporal summary
    fig2 = analyzer.plot_temporal_summary(
        time_column="timestamp",
        period="D",
        output_path="/tmp/temporal_summary.png"
    )
    print("✓ Temporal summary plot saved to /tmp/temporal_summary.png")

except Exception as e:
    print(f"⚠ Visualization skipped: {e}")

# Step 7: Save corpus with temporal analysis
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

read_data = ReadData(corpus=corpus)
read_data.write_corpus_to_json("/tmp/patient_temporal_analysis", corpus=corpus)
print("✓ Corpus saved to /tmp/patient_temporal_analysis")

# Step 8: Summary
print("\n" + "="*60)
print("ANALYSIS SUMMARY")
print("="*60)
print(f"Documents analyzed: {len(corpus.documents)}")
print(f"Sensor readings: {len(corpus.df)}")
print(f"Temporal links created: {linked_count}")
print(f"Time periods covered: {len(trend_df)} days")
print(f"Topics extracted: {sum(len(topics) for topics in topics.values())}")

print("\n✓ Temporal analysis complete!")
print("\nNext steps:")
print("  - Review temporal links in document metadata")
print("  - Analyze sentiment-sensor correlations")
print("  - Create temporal graph (requires topic modeling with keywords)")
print("  - Use MCP server for AI-assisted analysis")
