import pytest
import logging
from src.crisp_t.read_data import ReadData
from pkg_resources import resource_filename

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_cluster_initialization(corpus_fixture):
    from src.crisp_t.cluster import Cluster

    cluster = Cluster(corpus=corpus_fixture)
    assert cluster._corpus == corpus_fixture, "Corpus should be set correctly"
