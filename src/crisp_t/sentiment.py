import textacy.similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import operator
from tqdm import tqdm


from .model import Corpus
from .text import Text


class Sentiment:

    def __init__(self, corpus: Corpus):
        self._corpus = corpus
        self._analyzer = SentimentIntensityAnalyzer()
        self._text = Text(corpus)
        self._spacy_doc = self._text.make_spacy_doc()
        self._spacy_docs, self._ids = self._text.make_each_document_into_spacy_doc()
        self._id = self._corpus.id

    @property
    def corpus(self):
        return self._corpus

    def get_sentiment(self, documents=False, verbose=True):
        sentiment = {}
        if not documents:
            sentiment = {
                "sentiment": self._analyzer.polarity_scores(self._spacy_doc.text),
            }
            # add sentiment metadata to corpus
            self._corpus.metadata["sentiment"] = self.max_sentiment(
                sentiment["sentiment"]
            )
        else:
            for idx, doc in enumerate(self._spacy_docs):
                sentiment[str(self._ids[idx])] = self._analyzer.polarity_scores(
                    doc.text
                )
            documents_copy = []
            # add sentiment metadata to each document
            for doc in tqdm(self._corpus.documents, desc="Adding sentiment metadata", disable=len(self._corpus.documents) < 10):
                for idx, doc_id in enumerate(self._ids):
                    if doc.id == doc_id:
                        documents_copy.append(doc)
                        documents_copy[-1].metadata["sentiment"] = self.max_sentiment(
                            sentiment[str(doc_id)]
                        )
            self._corpus.documents = documents_copy

        if verbose:
            print("Sentiment Analysis Results:")
            for doc_id, sentiment_scores in sentiment.items():
                print(f"Document ID: {doc_id}")
                print(f"Sentiment Scores: {sentiment_scores}")
        return sentiment

    def max_sentiment(self, score):
        score.pop("compound", None)
        return max(score.items(), key=operator.itemgetter(1))[0]
