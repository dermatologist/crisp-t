"""
Dummy conftest.py for crisp_t.

If you don't know what this is for, just leave it empty.
Read more about conftest.py under:
- https://docs.pytest.org/en/stable/fixture.html
- https://docs.pytest.org/en/stable/writing_plugins.html
"""

import logging
import os
from pathlib import Path

import pytest

from src.crisp_t.csv import Csv
from src.crisp_t.read_data import ReadData

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def read_data_fixture():
    folder_path = str(Path(__file__).parent / "resources" / "")
    read_data = ReadData()
    read_data.read_source(folder_path)
    return read_data


@pytest.fixture
def corpus_fixture():
    folder_path = str(Path(__file__).parent / "resources" / "")
    read_data = ReadData()
    read_data.read_source(folder_path)
    corpus = read_data.create_corpus(
        name="Test Corpus", description="This is a test corpus"
    )
    return corpus


@pytest.fixture
def csv_fixture(corpus_fixture):
    return Csv(corpus=corpus_fixture)

@pytest.fixture
def folder_path_fixture():
    return str(Path(__file__).parent / "resources" / "")

@pytest.fixture
def csv_file_fixture():
    return str(Path(__file__).parent / "resources" / "food_coded.csv")


@pytest.fixture(autouse=True)
def run_before_and_after_tests(corpus_fixture):
    """Fixture to execute asserts before and after a test is run"""
    # Setup: fill with any logic you want

    yield  # this is where the testing happens

    # Teardown : fill with any logic you want
    corpus_fixture.pretty_print()
