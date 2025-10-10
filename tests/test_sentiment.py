import logging
from src.crisp_t.sentiment import Sentiment

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_sentiment_initialization(corpus_fixture):
    sentiment = Sentiment(corpus=corpus_fixture)
    assert sentiment._corpus == corpus_fixture, "Corpus should be set correctly"


def test_get_sentiment(corpus_fixture):
    sentiment = Sentiment(corpus=corpus_fixture)
    s = sentiment.get_sentiment(documents=True, verbose=False)
    print(s)
    doc1 = sentiment._corpus.documents[0].metadata["sentiment"]
    assert doc1 in ["neu", "pos", "neg"], "Sentiment should be one of 'neu', 'pos', or 'neg'"
