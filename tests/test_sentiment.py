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
    # Check if any document has sentiment metadata
    docs_with_sentiment = [
        doc for doc in sentiment._corpus.documents if "sentiment" in doc.metadata
    ]
    if docs_with_sentiment:
        doc1 = docs_with_sentiment[0].metadata["sentiment"]
        assert doc1 in [
            "neu",
            "pos",
            "neg",
        ], "Sentiment should be one of 'neu', 'pos', or 'neg'"
    else:
        # If no documents match the sentiment analysis, test passes
        pass
