"""
Copyright (C) 2025 Bell Eapen

This file is part of crisp-t.

crisp-t is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

crisp-t is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with crisp-t.  If not, see <https://www.gnu.org/licenses/>.
"""

import pandas as pd
import numpy as np
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from tabulate import tabulate
from typing import List, Dict, Any, Optional

from .model import Corpus
from .text import Text


class Cluster:

    def __init__(self, corpus: Optional[Corpus] = None):
        self._corpus = corpus
        self._ids = []
        self._lda_model: Optional[LdaModel] = None
        self._word2vec_model = None
        self._clusters = None
        self._clustered_data = None
        self._tsne_results = None
        self._kmeans_model = None
        self._processed_docs = None
        self._dictionary = None
        self._bag_of_words = None
        self._num_topics = 5
        self._passes = 15
        self.process()

    def process(self):
        """
        Process the data and perform clustering.
        """
        if self._corpus is None:
            raise ValueError("Corpus is not set")

        # Create a Text object
        text = Text(corpus=self._corpus)
        spacy_docs, ids = text.make_each_document_into_spacy_doc()
        self._ids = ids
        self._processed_docs = [
            self.tokenize(doc) for doc in spacy_docs if doc is not None
        ]
        self._dictionary = corpora.Dictionary(self._processed_docs)
        self._bag_of_words = [
            self._dictionary.doc2bow(doc) for doc in self._processed_docs
        ]

    def build_lda_model(self):
        if self._lda_model is None:
            self._lda_model = LdaModel(
                self._bag_of_words,
                num_topics=self._num_topics,
                id2word=self._dictionary,
                passes=self._passes,
            )
        return self._lda_model.show_topics(formatted=False)

    def print_topics(self, num_words=5, verbose=False):
        if self._lda_model is None:
            self.build_lda_model()
        if self._lda_model is None:
            raise ValueError("LDA model could not be built.")
        # Print the topics and their corresponding words
        # print(self._lda_model.print_topics(num_words=num_words))
        output = self._lda_model.print_topics(num_words=num_words)
        """ Output is like:
        [(0, '0.116*"category" + 0.093*"comparison" + 0.070*"incident" + 0.060*"theory" + 0.025*"Theory"'), (1, '0.040*"GT" + 0.026*"emerge" + 0.026*"pragmatic" + 0.026*"Barney" + 0.026*"contribution"'), (2, '0.084*"theory" + 0.044*"GT" + 0.044*"evaluation" + 0.024*"structure" + 0.024*"Glaser"'), (3, '0.040*"open" + 0.040*"QRMine" + 0.040*"coding" + 0.040*"category" + 0.027*"researcher"'), (4, '0.073*"coding" + 0.046*"structure" + 0.045*"GT" + 0.042*"Strauss" + 0.038*"Corbin"')]
        format this into human readable format as below:
        Topic 0: category(0.116), comparison(0.093), incident(0.070), theory(0.060), Theory(0.025)
        """
        if verbose:
            print("\nTopics: \n")
            for topic in output:
                topic_num = topic[0]
                topic_words = topic[1]
                words = []
                for word in topic_words.split("+"):
                    word = word.split("*")
                    words.append(f"{word[1].strip()}({word[0].strip()})")
                print(f"Topic {topic_num}: {', '.join(words)}")
        return output

    def tokenize(self, spacy_doc):
        return [
            token.lemma_
            for token in spacy_doc
            if not token.is_stop and not token.is_punct and not token.is_space
        ]

    def print_clusters(self, verbose=False):
        if self._lda_model is None:
            self.build_lda_model()
        if self._lda_model is None:
            raise ValueError("LDA model could not be built.")

        clusters = {}
        if verbose:
            print("\n Main topic in doc: \n")

        if self._processed_docs is None:
            raise ValueError("Processed documents are not available.")
        for i, doc in enumerate(
            self._processed_docs
        ):  # Changed from get_processed_docs() to _documents
            if self._dictionary is None:
                self._dictionary = corpora.Dictionary(self._processed_docs)
            bow = self._dictionary.doc2bow(doc)
            topic = self._lda_model.get_document_topics(bow)
            clusters[self._ids[i]] = topic

        if verbose:
            for doc_id, topic in clusters.items():
                print(f"Document ID: {doc_id}, Topic: {topic}")

        return clusters