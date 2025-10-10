import logging
from pathlib import Path

from src.crisp_t.csv import Csv

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_csv_initialization(corpus_fixture):
    csv = Csv(corpus=corpus_fixture)
    assert csv._corpus == corpus_fixture, "Corpus should be set correctly"


def test_read_csv(corpus_fixture, csv_file_fixture):
    csv = Csv(
        corpus=corpus_fixture,
        comma_separated_text_columns="comfort_food,comfort_food_reasons,diet_current",
        comma_separated_ignore_columns="type_sports,weight",
    )
    csv.read_csv(csv_file_fixture)
    print(csv.df.head())  # Print the first few rows of the DataFrame for debugging
    assert len(csv.df) > 0, "DataFrame should have rows after reading CSV"


def test_mark_missing(corpus_fixture, csv_file_fixture):
    csv = Csv(
        corpus=corpus_fixture,
        comma_separated_text_columns="comfort_food,comfort_food_reasons,diet_current",
    )
    csv.read_csv(csv_file_fixture)
    csv.mark_missing()
    print(csv.df.head())  # Print the first few rows of the DataFrame for debugging
    assert len(csv.df) > 0, "DataFrame should have rows after marking missing"


def test_read_xy(corpus_fixture, csv_file_fixture):
    csv = Csv(
        corpus=corpus_fixture,
        comma_separated_text_columns="comfort_food,comfort_food_reasons,diet_current",
    )
    csv.read_csv(csv_file_fixture)
    X, y = csv.read_xy(y="GPA")
    print(X, y)  # Print the first few rows of the DataFrame for debugging
    assert X is not None, "X should not be None"
    assert y is not None, "y should not be None"


def test_oversample(corpus_fixture, csv_file_fixture):
    csv = Csv(
        corpus=corpus_fixture,
        comma_separated_text_columns="comfort_food,comfort_food_reasons,diet_current",
    )
    csv.read_csv(csv_file_fixture)
    csv.drop_na()
    X, y = csv.read_xy(y="Gender")
    print(X, y)  # Print the first few rows of the DataFrame for debugging
    X_resampled, y_resampled = csv.oversample()  # type: ignore
    print(
        X_resampled, y_resampled
    )  # Print the first few rows of the DataFrame for debugging
    assert X_resampled is not None, "X_resampled should not be None"
