"""
Copyright (C) 2025 Bell Eapen

This file is part of crisp-t.

crisp-t is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

crisp-t is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with crisp-t.  If not, see <https://www.gnu.org/licenses/>.
"""

import logging
from datetime import datetime, timedelta

import pandas as pd

from .model import Corpus

logger = logging.getLogger(__name__)

# Common stop words for topic extraction
COMMON_STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "as",
    "is",
    "was",
    "are",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "should",
    "could",
    "may",
    "might",
    "can",
}


class TemporalAnalyzer:
    """
    Temporal analysis functionality for CRISP-T.
    Provides time-based linking, filtering, and analysis of text and numeric data.
    """

    def __init__(self, corpus: Corpus):
        """
        Initialize the TemporalAnalyzer with a corpus.

        Args:
            corpus: Corpus object to analyze.
        """
        self.corpus = corpus

    @staticmethod
    def parse_timestamp(timestamp_str: str | None) -> datetime | None:
        """
        Parse a timestamp string in various formats to datetime object.
        Returns None if parsing fails.
        """
        if not timestamp_str or pd.isna(timestamp_str):
            return None
        try:
            dt = pd.to_datetime(timestamp_str, errors="coerce")
            if pd.isna(dt):
                logger.warning(f"Failed to parse timestamp: {timestamp_str}")
                return None
            return dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else dt
        except Exception:
            logger.warning(f"Failed to parse timestamp: {timestamp_str}")
            return None

    def link_by_nearest_time(
        self, time_column: str = "timestamp", max_gap: timedelta | None = None
    ) -> Corpus:
        """
        Link documents to dataframe rows by nearest timestamp.

        Args:
            time_column: Name of the timestamp column in the dataframe.
            max_gap: Optional maximum time gap allowed for linking.

        Returns:
            Updated corpus with temporal links.
        """
        if self.corpus.df is None or time_column not in self.corpus.df.columns:
            raise ValueError(f"DataFrame does not have column '{time_column}'")

        # Parse dataframe timestamps
        df_times = self.corpus.df[time_column].apply(self.parse_timestamp)
        valid_df_indices = df_times.notna()

        # Update documents in corpus if metadata is changed
        updated_documents = []
        for doc in self.corpus.documents:
            if not getattr(doc, "timestamp", None):
                updated_documents.append(doc)
                continue
            doc_time = self.parse_timestamp(getattr(doc, "timestamp", None))
            if not doc_time:
                updated_documents.append(doc)
                continue
            time_diffs = (df_times[valid_df_indices] - doc_time).abs()
            if time_diffs.empty:
                updated_documents.append(doc)
                continue
            nearest_idx = time_diffs.idxmin()
            min_gap = time_diffs.min()
            if max_gap and min_gap > max_gap:
                updated_documents.append(doc)
                continue
            doc.metadata.setdefault("temporal_links", []).append(
                {
                    "df_index": int(nearest_idx),
                    "time_gap_seconds": min_gap.total_seconds(),
                    "link_type": "nearest_time",
                }
            )
            updated_documents.append(doc)

        self.corpus.documents = updated_documents
        return self.corpus

    def link_by_time_window(
        self,
        time_column: str = "timestamp",
        window_before: timedelta = timedelta(minutes=5),
        window_after: timedelta = timedelta(minutes=5),
    ) -> Corpus:
        """
        Link documents to all dataframe rows within a time window.

        Args:
            time_column: Name of the timestamp column in the dataframe.
            window_before: Time window before document timestamp.
            window_after: Time window after document timestamp.

        Returns:
            Updated corpus with temporal links.
        """
        if self.corpus.df is None or time_column not in self.corpus.df.columns:
            raise ValueError(f"DataFrame does not have column '{time_column}'")

        # Parse dataframe timestamps
        df_times = self.corpus.df[time_column].apply(self.parse_timestamp)
        valid_df_indices = df_times.notna()

        updated_documents = []
        for doc in self.corpus.documents:
            if not getattr(doc, "timestamp", None):
                updated_documents.append(doc)
                continue
            doc_time = self.parse_timestamp(getattr(doc, "timestamp", None))
            if not doc_time:
                updated_documents.append(doc)
                continue
            within_window = (df_times[valid_df_indices] >= doc_time - window_before) & (
                df_times[valid_df_indices] <= doc_time + window_after
            )
            matching_indices = df_times[valid_df_indices][within_window].index.tolist()
            if not matching_indices:
                updated_documents.append(doc)
                continue
            for idx in matching_indices:
                time_gap = (df_times[idx] - doc_time).total_seconds()
                doc.metadata.setdefault("temporal_links", []).append(
                    {
                        "df_index": int(idx),
                        "time_gap_seconds": time_gap,
                        "link_type": "time_window",
                    }
                )
            updated_documents.append(doc)

        self.corpus.documents = updated_documents
        return self.corpus

    def link_by_sequence(
        self,
        time_column: str = "timestamp",
        period: str = "W",  # Week by default
    ) -> Corpus:
        """
        Link documents and dataframe rows by time sequences (e.g., same week).

        Args:
            time_column: Name of the timestamp column in the dataframe.
            period: Pandas period string ('D' for day, 'W' for week, 'M' for month, 'Y' for year).

        Returns:
            Updated corpus with temporal links.
        """
        if self.corpus.df is None or time_column not in self.corpus.df.columns:
            raise ValueError(f"DataFrame does not have column '{time_column}'")

        # Parse dataframe timestamps
        df_times = self.corpus.df[time_column].apply(self.parse_timestamp)
        valid_df_indices = df_times.notna()
        df_times_valid = df_times[valid_df_indices]

        # Group dataframe rows by period
        df_periods = df_times_valid.dt.to_period(period)  # type: ignore[attr-defined]

        updated_documents = []
        for doc in self.corpus.documents:
            if not getattr(doc, "timestamp", None):
                updated_documents.append(doc)
                continue
            doc_time = self.parse_timestamp(getattr(doc, "timestamp", None))
            if not doc_time:
                updated_documents.append(doc)
                continue
            doc_period = pd.Period(doc_time, freq=period)
            matching_indices = df_periods[df_periods == doc_period].index.tolist()
            if not matching_indices:
                updated_documents.append(doc)
                continue
            for idx in matching_indices:
                doc.metadata.setdefault("temporal_links", []).append(
                    {
                        "df_index": int(idx),
                        "period": str(doc_period),
                        "link_type": "sequence",
                    }
                )
            updated_documents.append(doc)

        self.corpus.documents = updated_documents
        return self.corpus

    def filter_by_time_range(
        self,
        start_time: str | None = None,
        end_time: str | None = None,
        time_column: str = "timestamp",
        filter_documents: bool = True,
        filter_dataframe: bool = True,
    ) -> Corpus:
        """
        Filter documents and dataframe rows by time range.

        Args:
            start_time: Start time (inclusive) as ISO 8601 string.
            end_time: End time (inclusive) as ISO 8601 string.
            time_column: Name of the timestamp column in the dataframe.
            filter_documents: Whether to filter documents.
            filter_dataframe: Whether to filter dataframe rows.

        Returns:
            New corpus with filtered data.
        """
        start_dt = self.parse_timestamp(start_time) if start_time else None
        end_dt = self.parse_timestamp(end_time) if end_time else None

        filtered_documents = []
        if filter_documents:
            for doc in self.corpus.documents:
                doc_time = self.parse_timestamp(getattr(doc, "timestamp", None))
                if not getattr(doc, "timestamp", None) or not doc_time:
                    filtered_documents.append(doc)
                    continue
                if start_dt and doc_time < start_dt:
                    continue
                if end_dt and doc_time > end_dt:
                    continue
                filtered_documents.append(doc)
        else:
            filtered_documents = self.corpus.documents

        filtered_df = None
        if filter_dataframe and self.corpus.df is not None:
            if time_column in self.corpus.df.columns:
                df_times = self.corpus.df[time_column].apply(self.parse_timestamp)

                # Create filter mask
                mask = pd.Series(
                    [True] * len(self.corpus.df), index=self.corpus.df.index
                )

                if start_dt:
                    mask &= (df_times >= start_dt) | df_times.isna()
                if end_dt:
                    mask &= (df_times <= end_dt) | df_times.isna()

                filtered_df = self.corpus.df[mask].copy()
            else:
                filtered_df = self.corpus.df
        else:
            filtered_df = self.corpus.df

        # Create new corpus with filtered data
        new_corpus = Corpus(
            id=self.corpus.id,
            name=self.corpus.name,
            description=self.corpus.description,
            score=self.corpus.score,
            documents=filtered_documents,
            df=filtered_df,
            metadata=self.corpus.metadata.copy(),
            visualization=self.corpus.visualization.copy(),
        )

        return new_corpus

    def get_temporal_summary(
        self,
        time_column: str = "timestamp",
        period: str = "W",
        numeric_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Generate temporal summary of numeric and text data.

        Args:
            time_column: Name of the timestamp column in the dataframe.
            period: Pandas period string ('D' for day, 'W' for week, 'M' for month).
            numeric_columns: List of numeric columns to summarize. If None, uses all numeric columns.

        Returns:
            DataFrame with temporal summaries.
        """
        summary_data = []

        # Summarize dataframe data
        if self.corpus.df is not None and time_column in self.corpus.df.columns:
            df_times = self.corpus.df[time_column].apply(self.parse_timestamp)
            valid_mask = df_times.notna()

            if valid_mask.any():
                df_with_times = self.corpus.df[valid_mask].copy()
                df_with_times["_period"] = df_times[valid_mask].dt.to_period(period)  # type: ignore[attr-defined]

                # Select numeric columns
                if numeric_columns is None:
                    numeric_columns = df_with_times.select_dtypes(
                        include=["number"]
                    ).columns.tolist()

                # Group by period and aggregate
                if numeric_columns:
                    grouped = df_with_times.groupby("_period")[numeric_columns].agg(
                        ["mean", "std", "min", "max", "count"]
                    )
                    summary_data.append(grouped)

        # Summarize document data
        doc_counts = {}
        for doc in self.corpus.documents:
            if not doc.timestamp:
                continue

            doc_time = self.parse_timestamp(doc.timestamp)
            if not doc_time:
                continue

            doc_period = pd.Period(doc_time, freq=period)
            doc_counts[doc_period] = doc_counts.get(doc_period, 0) + 1

        if doc_counts:
            doc_summary = pd.DataFrame(
                list(doc_counts.items()), columns=["period", "document_count"]
            ).set_index("period")
            summary_data.append(doc_summary)

        if not summary_data:
            return pd.DataFrame()

        # Combine summaries
        if len(summary_data) == 1:
            return summary_data[0]
        else:
            return pd.concat(summary_data, axis=1)

    def add_temporal_relationship(
        self,
        doc_id: str,
        df_column: str,
        relation: str = "temporal_correlation",
        time_column: str = "timestamp",
    ):
        """
        Add temporal relationship between document and dataframe column.

        Args:
            doc_id: Document ID.
            df_column: DataFrame column name.
            relation: Type of temporal relationship.
            time_column: Name of timestamp column in dataframe.
        """
        doc = self.corpus.get_document_by_id(doc_id)
        if not doc or not doc.timestamp:
            raise ValueError(f"Document {doc_id} not found or has no timestamp")

        if self.corpus.df is None or df_column not in self.corpus.df.columns:
            raise ValueError(f"DataFrame column {df_column} not found")

        # Add relationship to corpus metadata
        self.corpus.add_relationship(
            first=f"text:{doc_id}",
            second=f"numb:{df_column}",
            relation=relation,
        )

    def get_temporal_sentiment_trend(
        self,
        period: str = "D",
        aggregation: str = "mean",
    ) -> pd.DataFrame:
        """
        Analyze sentiment trends over time.
        Requires documents to have sentiment metadata and timestamps.

        Args:
            period: Pandas period string ('D' for day, 'W' for week, 'M' for month).
            aggregation: Aggregation method ('mean', 'median', 'max', 'min').

        Returns:
            DataFrame with sentiment trends over time.
        """
        sentiment_data = []

        for doc in self.corpus.documents:
            if not doc.timestamp or "sentiment" not in doc.metadata:
                continue

            doc_time = self.parse_timestamp(doc.timestamp)
            if not doc_time:
                continue

            # Convert sentiment to numeric score
            sentiment = doc.metadata["sentiment"]
            score_map = {
                "pos": 1.0,
                "positive": 1.0,
                "neg": -1.0,
                "negative": -1.0,
                "neu": 0.0,
                "neutral": 0.0,
            }
            sentiment_score = score_map.get(sentiment.lower(), 0.0)

            sentiment_data.append(
                {
                    "timestamp": doc_time,
                    "period": pd.Period(doc_time, freq=period),
                    "sentiment_score": sentiment_score,
                }
            )

        if not sentiment_data:
            return pd.DataFrame()

        df = pd.DataFrame(sentiment_data)

        # Aggregate by period
        if aggregation == "mean":
            result = df.groupby("period")["sentiment_score"].mean()
        elif aggregation == "median":
            result = df.groupby("period")["sentiment_score"].median()
        elif aggregation == "max":
            result = df.groupby("period")["sentiment_score"].max()
        elif aggregation == "min":
            result = df.groupby("period")["sentiment_score"].min()
        else:
            result = df.groupby("period")["sentiment_score"].mean()

        result_df = result.to_frame()
        result_df["document_count"] = df.groupby("period").size()

        return result_df

    def get_temporal_topics(
        self,
        period: str = "W",
        top_n: int = 5,
    ) -> dict[str, list[str]]:
        """
        Extract topics over time periods.
        Requires documents to have timestamp and optionally topics metadata.

        Args:
            period: Pandas period string ('D' for day, 'W' for week, 'M' for month).
            top_n: Number of top keywords/topics per period.

        Returns:
            Dictionary mapping period to list of top topics/keywords.
        """
        from collections import Counter

        period_topics = {}

        # Group documents by period
        period_docs = {}
        for doc in self.corpus.documents:
            if not doc.timestamp:
                continue

            doc_time = self.parse_timestamp(doc.timestamp)
            if not doc_time:
                continue

            doc_period = str(pd.Period(doc_time, freq=period))
            if doc_period not in period_docs:
                period_docs[doc_period] = []
            period_docs[doc_period].append(doc)

        # Extract topics for each period
        for period_key, docs in period_docs.items():
            # If documents have topic metadata, aggregate them
            if any("topics" in doc.metadata for doc in docs):
                all_topics = []
                for doc in docs:
                    if "topics" in doc.metadata:
                        topics = doc.metadata["topics"]
                        if isinstance(topics, list):
                            all_topics.extend(topics)
                        elif isinstance(topics, str):
                            all_topics.append(topics)

                # Count and get top topics
                topic_counts = Counter(all_topics)
                period_topics[period_key] = [
                    topic for topic, _ in topic_counts.most_common(top_n)
                ]
            else:
                # Simple keyword extraction from text (fallback)
                all_text = " ".join(doc.text for doc in docs)
                words = all_text.lower().split()
                # Filter out common stop words and get most common
                filtered_words = [
                    w for w in words if w not in COMMON_STOP_WORDS and len(w) > 3
                ]
                word_counts = Counter(filtered_words)
                period_topics[period_key] = [
                    word for word, _ in word_counts.most_common(top_n)
                ]

        return period_topics

    def plot_temporal_sentiment(
        self,
        period: str = "D",
        aggregation: str = "mean",
        output_path: str | None = None,
    ):
        """
        Plot sentiment trends over time.

        Args:
            period: Pandas period string ('D' for day, 'W' for week, 'M' for month).
            aggregation: Aggregation method ('mean', 'median', 'max', 'min').
            output_path: Optional path to save the plot.

        Returns:
            Matplotlib figure object.
        """
        import matplotlib.pyplot as plt

        trend_df = self.get_temporal_sentiment_trend(
            period=period, aggregation=aggregation
        )

        if trend_df.empty:
            raise ValueError("No temporal sentiment data available")

        fig, ax = plt.subplots(figsize=(12, 6))

        # Convert period index to string for plotting
        x_labels = [str(p) for p in trend_df.index]
        x_pos = range(len(x_labels))

        ax.plot(
            x_pos, trend_df["sentiment_score"], marker="o", linewidth=2, markersize=8
        )
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_xlabel("Time Period")
        ax.set_ylabel("Sentiment Score")
        ax.set_title(f"Temporal Sentiment Trend ({period} periods, {aggregation})")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)

        return fig

    def plot_temporal_summary(
        self,
        time_column: str = "timestamp",
        period: str = "W",
        numeric_columns: list[str] | None = None,
        output_path: str | None = None,
    ):
        """
        Plot temporal summary of numeric data.

        Args:
            time_column: Name of the timestamp column in the dataframe.
            period: Pandas period string ('D' for day, 'W' for week, 'M' for month).
            numeric_columns: List of numeric columns to plot.
            output_path: Optional path to save the plot.

        Returns:
            Matplotlib figure object.
        """
        import matplotlib.pyplot as plt

        summary_df = self.get_temporal_summary(
            time_column=time_column, period=period, numeric_columns=numeric_columns
        )

        if summary_df.empty:
            raise ValueError("No temporal summary data available")

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        x_labels = [str(p) for p in summary_df.index]
        x_pos = range(len(x_labels))

        # Plot document count over time
        if "document_count" in summary_df.columns:
            axes[0].bar(x_pos, summary_df["document_count"], alpha=0.7)
            axes[0].set_xticks(x_pos)
            axes[0].set_xticklabels(x_labels, rotation=45, ha="right")
            axes[0].set_xlabel("Time Period")
            axes[0].set_ylabel("Document Count")
            axes[0].set_title("Document Count Over Time")
            axes[0].grid(True, alpha=0.3)

        # Plot numeric data trends
        if isinstance(summary_df.columns, pd.MultiIndex):
            # MultiIndex columns from numeric aggregations
            for col in summary_df.columns.get_level_values(0).unique():
                if col != "document_count":
                    mean_col = (col, "mean")
                    if mean_col in summary_df.columns:
                        axes[1].plot(x_pos, summary_df[mean_col], marker="o", label=col)

            axes[1].set_xticks(x_pos)
            axes[1].set_xticklabels(x_labels, rotation=45, ha="right")
            axes[1].set_xlabel("Time Period")
            axes[1].set_ylabel("Mean Value")
            axes[1].set_title("Numeric Data Trends Over Time")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)

        return fig
