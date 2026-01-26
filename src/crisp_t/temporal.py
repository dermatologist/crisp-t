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

from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict, Any
import pandas as pd
from .model import Corpus, Document


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
    def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
        """
        Parse a timestamp string in various formats to datetime object.

        Args:
            timestamp_str: Timestamp string to parse.

        Returns:
            datetime object if parsing succeeds, None otherwise.
        """
        if not timestamp_str or pd.isna(timestamp_str):
            return None

        # Common timestamp formats
        formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",  # ISO 8601 with milliseconds
            "%Y-%m-%dT%H:%M:%SZ",     # ISO 8601
            "%Y-%m-%dT%H:%M:%S",      # ISO 8601 without Z
            "%Y-%m-%d %H:%M:%S",      # Common format
            "%Y-%m-%d",               # Date only
            "%m/%d/%Y %H:%M:%S",      # US format with time
            "%m/%d/%Y",               # US date only
            "%d/%m/%Y %H:%M:%S",      # European format with time
            "%d/%m/%Y",               # European date only
        ]

        for fmt in formats:
            try:
                return datetime.strptime(str(timestamp_str), fmt)
            except ValueError:
                continue

        # Try pandas parser as fallback
        try:
            return pd.to_datetime(timestamp_str)
        except Exception:
            return None

    def link_by_nearest_time(
        self, time_column: str = "timestamp", max_gap: Optional[timedelta] = None
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

        for doc in self.corpus.documents:
            if not doc.timestamp:
                continue

            doc_time = self.parse_timestamp(doc.timestamp)
            if not doc_time:
                continue

            # Find nearest row
            time_diffs = (df_times[valid_df_indices] - doc_time).abs()
            if len(time_diffs) == 0:
                continue

            nearest_idx = time_diffs.idxmin()
            min_gap = time_diffs.min()

            # Check max gap if specified
            if max_gap and min_gap > max_gap:
                continue

            # Store link in document metadata
            if "temporal_links" not in doc.metadata:
                doc.metadata["temporal_links"] = []

            doc.metadata["temporal_links"].append(
                {
                    "df_index": int(nearest_idx),
                    "time_gap_seconds": min_gap.total_seconds(),
                    "link_type": "nearest_time",
                }
            )

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

        for doc in self.corpus.documents:
            if not doc.timestamp:
                continue

            doc_time = self.parse_timestamp(doc.timestamp)
            if not doc_time:
                continue

            # Find all rows within window
            within_window = (
                (df_times[valid_df_indices] >= doc_time - window_before)
                & (df_times[valid_df_indices] <= doc_time + window_after)
            )

            matching_indices = df_times[valid_df_indices][within_window].index.tolist()

            if not matching_indices:
                continue

            # Store links in document metadata
            if "temporal_links" not in doc.metadata:
                doc.metadata["temporal_links"] = []

            for idx in matching_indices:
                time_gap = (df_times[idx] - doc_time).total_seconds()
                doc.metadata["temporal_links"].append(
                    {
                        "df_index": int(idx),
                        "time_gap_seconds": time_gap,
                        "link_type": "time_window",
                    }
                )

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
        df_periods = df_times_valid.dt.to_period(period)

        for doc in self.corpus.documents:
            if not doc.timestamp:
                continue

            doc_time = self.parse_timestamp(doc.timestamp)
            if not doc_time:
                continue

            doc_period = pd.Period(doc_time, freq=period)

            # Find all rows in the same period
            matching_indices = df_periods[df_periods == doc_period].index.tolist()

            if not matching_indices:
                continue

            # Store links in document metadata
            if "temporal_links" not in doc.metadata:
                doc.metadata["temporal_links"] = []

            for idx in matching_indices:
                doc.metadata["temporal_links"].append(
                    {
                        "df_index": int(idx),
                        "period": str(doc_period),
                        "link_type": "sequence",
                    }
                )

        return self.corpus

    def filter_by_time_range(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
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
                if not doc.timestamp:
                    # Include documents without timestamps
                    filtered_documents.append(doc)
                    continue

                doc_time = self.parse_timestamp(doc.timestamp)
                if not doc_time:
                    filtered_documents.append(doc)
                    continue

                # Check time range
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
                mask = pd.Series([True] * len(self.corpus.df), index=self.corpus.df.index)

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
        numeric_columns: Optional[List[str]] = None,
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
                df_with_times["_period"] = df_times[valid_mask].dt.to_period(period)

                # Select numeric columns
                if numeric_columns is None:
                    numeric_columns = df_with_times.select_dtypes(include=["number"]).columns.tolist()

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
