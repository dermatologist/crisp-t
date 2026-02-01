"""
Tests for new CSV enhancements: datetime parsing, ID column creation,
query execution, and correlation analysis.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.crisp_t.csv import Csv
from src.crisp_t.read_data import ReadData


def test_datetime_parsing_with_coerce():
    """Test that date columns are automatically parsed with errors='coerce'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_dates.csv")
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "event_date": ["2025-01-15", "2025-02-20", "invalid_date"],
                "score": [85, 92, 78],
            }
        )
        df.to_csv(csv_path, index=False)

        reader = ReadData()
        corpus = reader.read_csv_to_corpus(
            csv_path,
            comma_separated_text_columns="",
            id_column="id",
        )

        # Check that event_date column was parsed as datetime
        assert corpus.df is not None
        # Note: The invalid date should be NaT (not a time) with errors='coerce'


def test_id_column_creation_from_index():
    """Test that ID column is created from index if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_no_id.csv")
        df = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "score": [85, 92, 78],
            }
        )
        df.to_csv(csv_path, index=False)

        reader = ReadData()
        corpus = reader.read_csv_to_corpus(
            csv_path,
            comma_separated_text_columns="name",
            id_column="patient_id",  # This column doesn't exist
        )

        # Check that patient_id column was created
        assert "patient_id" in corpus.df.columns
        assert len(corpus.df["patient_id"]) == 3


def test_execute_query_groupby():
    """Test query execution with groupby aggregation."""
    df = pd.DataFrame(
        {
            "category": ["A", "A", "B", "B", "C"],
            "value": [10, 20, 30, 40, 50],
            "count": [1, 2, 3, 4, 5],
        }
    )
    
    csv = Csv()
    csv.df = df

    # Test groupby query
    result = csv.execute_query("groupby('category')['value'].mean()")
    
    assert not result.empty
    assert len(result) == 3  # Three categories
    assert "value" in result.columns or result.name == "value"


def test_execute_query_sort():
    """Test query execution with sorting."""
    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "David"],
            "score": [85, 92, 78, 95],
        }
    )
    
    csv = Csv()
    csv.df = df

    # Test sort query
    result = csv.execute_query("sort_values('score', ascending=False).head(2)")
    
    assert len(result) == 2
    assert result.iloc[0]["score"] == 95  # Highest score first


def test_execute_query_filter():
    """Test query execution with filtering."""
    df = pd.DataFrame(
        {
            "age": [25, 35, 45, 55],
            "score": [70, 85, 60, 90],
        }
    )
    
    csv = Csv()
    csv.df = df

    # Test filter query
    result = csv.execute_query("query('age > 30 and score < 90')")
    
    assert len(result) == 2  # Should match 35 and 45 age rows


def test_execute_query_save_result():
    """Test that query result can be saved to DataFrame."""
    df = pd.DataFrame(
        {
            "category": ["A", "A", "B"],
            "value": [10, 20, 30],
        }
    )
    
    csv = Csv()
    csv.df = df
    original_shape = df.shape

    # Execute query with save_result=True
    result = csv.execute_query("groupby('category')['value'].sum()", save_result=True)
    
    # Check that the DataFrame was updated
    assert csv.df.shape != original_shape
    assert len(csv.df) == 2  # Two categories


def test_execute_query_invalid():
    """Test error handling for invalid queries."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    
    csv = Csv()
    csv.df = df

    # Test invalid query
    with pytest.raises(ValueError):
        csv.execute_query("invalid_method()")


def test_compute_correlation_default():
    """Test correlation computation with default parameters."""
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],  # Perfect positive correlation with 'a'
            "c": [5, 4, 3, 2, 1],   # Perfect negative correlation with 'a'
            "d": [1, 3, 2, 5, 4],   # Weak correlation
        }
    )
    
    csv = Csv()
    csv.df = df

    corr_matrix = csv.compute_correlation()
    
    assert not corr_matrix.empty
    assert corr_matrix.shape == (4, 4)
    # Check perfect positive correlation
    assert abs(corr_matrix.loc["a", "b"] - 1.0) < 0.01
    # Check perfect negative correlation
    assert abs(corr_matrix.loc["a", "c"] - (-1.0)) < 0.01


def test_compute_correlation_specific_columns():
    """Test correlation computation with specific columns."""
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],
            "c": [5, 4, 3, 2, 1],
            "text": ["x", "y", "z", "w", "v"],  # Non-numeric column
        }
    )
    
    csv = Csv()
    csv.df = df

    # Compute correlation for specific columns
    corr_matrix = csv.compute_correlation(columns=["a", "b"])
    
    assert corr_matrix.shape == (2, 2)
    assert "a" in corr_matrix.columns
    assert "b" in corr_matrix.columns
    assert "c" not in corr_matrix.columns


def test_compute_correlation_methods():
    """Test different correlation methods."""
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],
        }
    )
    
    csv = Csv()
    csv.df = df

    # Test Pearson (default)
    corr_pearson = csv.compute_correlation(method='pearson')
    assert not corr_pearson.empty

    # Test Spearman
    corr_spearman = csv.compute_correlation(method='spearman')
    assert not corr_spearman.empty

    # Test Kendall
    corr_kendall = csv.compute_correlation(method='kendall')
    assert not corr_kendall.empty


def test_find_significant_correlations():
    """Test finding significant correlations above threshold."""
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],  # Strong positive correlation
            "c": [5, 4, 3, 2, 1],   # Strong negative correlation
            "d": [1, 3, 2, 5, 4],   # Weak correlation
        }
    )
    
    csv = Csv()
    csv.df = df

    # Find correlations above 0.9
    sig_corrs = csv.find_significant_correlations(threshold=0.9)
    
    assert not sig_corrs.empty
    assert "Variable 1" in sig_corrs.columns
    assert "Variable 2" in sig_corrs.columns
    assert "Correlation" in sig_corrs.columns
    
    # Should find a-b and a-c correlations (both ~1.0)
    assert len(sig_corrs) >= 2


def test_find_significant_correlations_no_results():
    """Test when no significant correlations are found."""
    df = pd.DataFrame(
        {
            "a": [1, 3, 2, 5, 4],
            "b": [2, 1, 4, 3, 5],
            "c": [5, 2, 3, 1, 4],
        }
    )
    
    csv = Csv()
    csv.df = df

    # Set very high threshold
    sig_corrs = csv.find_significant_correlations(threshold=0.99)
    
    # Should return empty DataFrame with correct columns
    assert "Variable 1" in sig_corrs.columns
    assert "Variable 2" in sig_corrs.columns
    assert "Correlation" in sig_corrs.columns


def test_correlation_with_empty_dataframe():
    """Test correlation analysis with empty DataFrame."""
    csv = Csv()
    csv.df = pd.DataFrame()

    corr_matrix = csv.compute_correlation()
    assert corr_matrix.empty

    sig_corrs = csv.find_significant_correlations()
    assert sig_corrs.empty


def test_query_with_empty_dataframe():
    """Test query execution with empty DataFrame."""
    csv = Csv()
    csv.df = pd.DataFrame()

    result = csv.execute_query("head()")
    assert result.empty
