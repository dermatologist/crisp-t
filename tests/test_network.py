import logging
from src.crisp_t.network import Network

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_network_initialization(corpus_fixture):
    network = Network(corpus_fixture)
    assert network is not None, "Network should be initialized"


def test_cooccurrence_network(corpus_fixture):
    network = Network(corpus_fixture)
    graph = network.cooccurence_network(window_size=2)
    print(graph)
    assert graph is not None, "Co-occurrence network should be created"
    assert len(graph.nodes) > 0, "Graph should have nodes"
    assert len(graph.edges) > 0, "Graph should have edges"


def test_graph_as_dict(corpus_fixture):
    network = Network(corpus_fixture)
    graph = network.cooccurence_network(window_size=2)
    graph_dict = network.graph_as_dict()
    print(graph_dict)
    assert graph_dict is not None, "Graph dictionary should be created"


def test_similarity_network(corpus_fixture):
    network = Network(corpus_fixture)
    graph = network.similarity_network(method="levenshtein")
    print(graph)
    assert graph is not None, "Similarity network should be created"
    assert len(graph.nodes) > 0, "Graph should have nodes"
    assert len(graph.edges) > 0, "Graph should have edges"
    graph_dict = network.graph_as_dict()
    print(graph_dict)
    assert graph_dict is not None, "Graph dictionary should be created"
