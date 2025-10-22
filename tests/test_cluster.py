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
    topics = cluster.print_topics(num_words=8)
    assert (
        len(topics) == cluster._num_topics
    ), "Number of topics should match the specified number"


def test_print_topics_default_num_words(corpus_fixture):
    """Test that default num_words is 8"""
    cluster = Cluster(corpus=corpus_fixture)
    cluster.build_lda_model()
    # Call without arguments to test default
    topics = cluster.print_topics()
    assert (
        len(topics) == cluster._num_topics
    ), "Number of topics should match the specified number"
    # Check that lda_raw_output is stored in metadata
    assert "lda_raw_output" in corpus_fixture.metadata, "lda_raw_output should be in corpus metadata"


def test_num_topics_default_is_eight(corpus_fixture):
    """Test that default num_topics is 8 as per Mettler et al. (2025)"""
    cluster = Cluster(corpus=corpus_fixture)
    assert cluster._num_topics == 8, "Default num_topics should be 8"


def test_print_clusters(corpus_fixture):
    cluster = Cluster(corpus=corpus_fixture)
    cluster.build_lda_model()
    clusters = cluster.print_clusters(verbose=True)
    # clusters is a dictionary mapping document IDs to their topic assignments
    # so the length is the number of documents, not topics
    assert (
        len(clusters) == len(corpus_fixture.documents)
    ), "Number of cluster assignments should match the number of documents"


def test_coherence_and_perplexity_in_metadata(corpus_fixture):
    """Test that coherence and perplexity scores are added to corpus metadata"""
    cluster = Cluster(corpus=corpus_fixture)
    cluster.build_lda_model()
    cluster.print_clusters(verbose=False)
    
    assert "coherence_score" in corpus_fixture.metadata, "Coherence score should be in corpus metadata"
    assert "perplexity" in corpus_fixture.metadata, "Perplexity should be in corpus metadata"
    
    # Check that values are numeric
    assert isinstance(corpus_fixture.metadata["coherence_score"], float), "Coherence score should be a float"
    assert isinstance(corpus_fixture.metadata["perplexity"], float), "Perplexity should be a float"


def test_format_topics_sentences(corpus_fixture):
    cluster = Cluster(corpus=corpus_fixture)
    cluster.build_lda_model()
    topics = cluster.print_topics(num_words=8)
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
