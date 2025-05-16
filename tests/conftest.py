"""
    Dummy conftest.py for crisp_t.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""

import os
import pytest
import logging
from src.crisp_t.read_data import ReadData
from pkg_resources import resource_filename

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def read_data_fixture():
    folder_path = resource_filename("src.crisp_t.resources", os.sep)
    read_data = ReadData()
    read_data.read_source(folder_path)
    return read_data


@pytest.fixture
def corpus_fixture():
    folder_path = resource_filename("src.crisp_t.resources", os.sep)
    read_data = ReadData()
    read_data.read_source(folder_path)
    corpus = read_data.create_corpus(
        name="Test Corpus", description="This is a test corpus"
    )
    return corpus
