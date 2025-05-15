import pytest
import logging
from src.crisp_t.read_data import ReadData
from pkg_resources import resource_filename
from src.crisp_t.cluster import Cluster

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_cluster_initialization(corpus_fixture):
    cluster = Cluster(corpus=corpus_fixture)
    assert cluster._corpus == corpus_fixture, "Corpus should be set correctly"


def test_build_lda_model(corpus_fixture):

    cluster = Cluster(corpus=corpus_fixture)
    cluster.build_lda_model()
    assert cluster._lda_model is not None, "LDA model should be built"
    assert (
        cluster._lda_model.num_topics == cluster._num_topics
    ), "Number of topics in LDA model should match the specified number"

def test_print_topics(corpus_fixture):
    cluster = Cluster(corpus=corpus_fixture)
    cluster.build_lda_model()
    topics = cluster.print_topics(num_words=5, verbose=False)
    assert len(topics) == cluster._num_topics, "Number of topics should match the specified number"

def test_print_clusters(corpus_fixture):
    cluster = Cluster(corpus=corpus_fixture)
    cluster.build_lda_model()
    clusters = cluster.print_clusters(verbose=True)
    assert len(clusters) == cluster._num_topics, "Number of clusters should match the specified number"