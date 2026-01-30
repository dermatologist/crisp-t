import operator

from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .model import Corpus
from .text import Text


class Sentiment:

    def __init__(self, corpus: Corpus):
        self._corpus = corpus
        self._analyzer = SentimentIntensityAnalyzer()
        self._text = Text(corpus)
        self._spacy_docs, self._ids = self._text.make_each_document_into_spacy_doc()
        self._spacy_doc = None  # Lazy initialization
        self._id = self._corpus.id

    @property
    def corpus(self):
        return self._corpus

    def get_sentiment(self, documents=False, verbose=True):
        sentiment = {}
        if not documents:
            if self._spacy_doc is None:
                self._spacy_doc = self._text.make_spacy_doc()
            sentiment = {
                "sentiment": self._analyzer.polarity_scores(self._spacy_doc.text),
            }
            # add sentiment metadata to corpus
            self._corpus.metadata["sentiment"] = self.max_sentiment(
                sentiment["sentiment"]
            )
        else:
            sentiment = {
                str(self._ids[idx]): self._analyzer.polarity_scores(doc.text)
                for idx, doc in enumerate(self._spacy_docs)
            }
            # add sentiment metadata to matching documents only (in-place)
            for doc in tqdm(
                self._corpus.documents,
                desc="Adding sentiment metadata",
                disable=len(self._corpus.documents) < 10,
            ):
                doc_id_str = str(doc.id)
                if doc_id_str in sentiment:
                    doc.metadata["sentiment"] = self.max_sentiment(
                        sentiment[doc_id_str]
                    )

        if verbose:
            print("Sentiment Analysis Results:")
            for doc_id, sentiment_scores in sentiment.items():
                print(f"Document ID: {doc_id}")
                print(f"Sentiment Scores: {sentiment_scores}")
        return sentiment

    def max_sentiment(self, score):
        return max(
            ((k, v) for k, v in score.items() if k != "compound"),
            key=operator.itemgetter(1),
        )[0]
