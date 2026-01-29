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

import textacy.representations.network as network

from .cluster import Cluster
from .model import Corpus
from .text import Text


class Network:
    """
    A class to represent a network of documents and their relationships.
    """

    def __init__(self, corpus: Corpus):
        """
        Initialize the Network with a corpus.

        :param corpus: Corpus object containing documents to be included in the network.
        """
        self._corpus = corpus
        self._cluster = Cluster(corpus)
        self._processed_docs = self._cluster.processed_docs
        self._graph = None

    def cooccurence_network(self, window_size=2):
        self._graph = network.build_cooccurrence_network(
            self._processed_docs, window_size=window_size
        )
        return self._graph

    def similarity_network(self, method="levenshtein"):
        text = Text(self._corpus)
        docs = text.make_spacy_doc()
        data = [sent.text.lower() for sent in docs.sents]
        self._graph = network.build_similarity_network(data, method)
        return self._graph

    def graph_as_dict(self):
        """
        Convert the graph to a dictionary representation.

        :return: Dictionary representation of the graph.
        """
        if self._graph is None:
            raise ValueError(
                "Graph has not been created yet. Call cooccurence_network() first."
            )
        return sorted(self._graph.adjacency())[0]
