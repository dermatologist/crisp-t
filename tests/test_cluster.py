import logging
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
    topics = cluster.print_topics(num_words=5)
    assert (
        len(topics) == cluster._num_topics
    ), "Number of topics should match the specified number"


def test_print_clusters(corpus_fixture):
    cluster = Cluster(corpus=corpus_fixture)
    cluster.build_lda_model()
    clusters = cluster.print_clusters(verbose=True)
    assert (
        len(clusters) == cluster._num_topics
    ), "Number of clusters should match the specified number"


def test_format_topics_sentences(corpus_fixture):
    cluster = Cluster(corpus=corpus_fixture)
    cluster.build_lda_model()
    topics = cluster.print_topics(num_words=5)
    pandas_df = cluster.format_topics_sentences(topics)
    # print pandas dataframe using tabulate
    print(pandas_df.head())
    assert pandas_df is not None, "Formatted topics sentences should not be None"


def test_most_representative_docs(corpus_fixture):
    cluster = Cluster(corpus=corpus_fixture)
    cluster.build_lda_model()
    most_representative_docs = cluster.most_representative_docs()
    print(most_representative_docs.head())
    assert (
        most_representative_docs is not None
    ), "Most representative documents should not be None"


def test_topics_per_document(corpus_fixture):
    cluster = Cluster(corpus=corpus_fixture)
    cluster.build_lda_model()
    (dominant_topics, topic_percentages) = cluster.topics_per_document()
    print(dominant_topics, topic_percentages)
    assert dominant_topics is not None, "Dominant topics should not be None"
