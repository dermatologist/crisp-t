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

import logging

import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import Word2Vec
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tabulate import tabulate
from tqdm import tqdm

from .model import Corpus
from .text import Text

# Set the logging level for the 'gensim' logger
logging.getLogger("gensim").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class Cluster:

    def __init__(self, corpus: Corpus):
        self._corpus = corpus
        self._ids = []
        self._lda_model: LdaModel | None = None
        self._word2vec_model = None
        self._clusters = None
        self._clustered_data = None
        self._tsne_results = None
        self._kmeans_model = None
        self._processed_docs = None
        self._dictionary = None
        self._bag_of_words = None
        # Mettler et al. (2025) find 8 as the optimal value for num_topics.
        # Their research recommends LDA over HDP and LSI for topic modeling.
        # The sensitivity analysis of coherence and perplexity indicates that organizing
        # the subsequent detailed analysis around 8 topics may be a good approach.
        self._num_topics = 8
        self._passes = 15
        self.process()

    @property
    def processed_docs(self) -> list[list[str]]:
        """
        Get the processed documents.

        :return: List of processed documents.
        """
        if self._processed_docs is None:
            raise ValueError("Processed documents are not available.")
        return self._processed_docs

    def process(self):
        """
        Process the data and perform clustering.
        """
        if self._corpus is None:
            raise ValueError("Corpus is not set")

        # Create a Text object
        text = Text(corpus=self._corpus)
        spacy_docs, ids = text.make_each_document_into_spacy_doc(self._corpus.id)
        self._ids = ids
        self._processed_docs = [
            self.tokenize(doc) for doc in spacy_docs if doc is not None
        ]
        self._dictionary = corpora.Dictionary(self._processed_docs)
        self._bag_of_words = [
            self._dictionary.doc2bow(doc) for doc in self._processed_docs
        ]
        # self._corpus.metadata["bag_of_words"] = self._bag_of_words

    def build_lda_model(self, topics: int = 0):
        if self._lda_model is None:
            logger.info("Building LDA model... This may take a while.")
            self._lda_model = LdaModel(
                self._bag_of_words,
                num_topics=topics if topics != 0 else self._num_topics,
                id2word=self._dictionary,
                passes=self._passes,
            )
        _word_cloud = self._lda_model.show_topics(formatted=False)

        # Sanitize _word_cloud for JSON serialization
        def _safe_topic(topic):
            topic_num, words = topic
            safe_words = [(str(w), float(p)) for w, p in words]
            return [int(topic_num), safe_words]

        safe_word_cloud = [_safe_topic(topic) for topic in _word_cloud]
        if self._corpus is not None:
            logger.info("Storing word cloud in corpus metadata")
            self._corpus.visualization["word_cloud"] = safe_word_cloud
        return _word_cloud

    def print_topics(self, num_words=8):
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
        topics = ""
        print("\nTopics: \n")
        for topic in output:
            topic_num = topic[0]
            topic_words = topic[1]
            words = []
            for word in topic_words.split("+"):
                word = word.split("*")
                words.append(f"{word[1].strip()}({word[0].strip()})")
            print(f"Topic {topic_num}: {', '.join(words)}")
            topics += f"Topic {topic_num}: {', '.join(words)}\n"
        self._corpus.metadata["topics"] = topics
        # Store raw LDA output for visualization
        self._corpus.metadata["lda_raw_output"] = [(int(topic[0]), topic[1]) for topic in output]
        return output

    def tokenize(self, spacy_doc):
        return [
            token.lemma_
            for token in spacy_doc
            if not token.is_stop and not token.is_punct and not token.is_space
        ]

    def _json_safe(self, obj):
        """
        Convert numpy/pandas objects and dtypes (e.g., np.int64, np.float32, ndarray,
        DataFrame, Series) into JSON-serializable Python built-in types.
        """
        # Local imports to avoid top-level hard dependencies for type checking
        try:
            import numpy as _np
        except Exception:  # pragma: no cover - numpy is already a dependency
            _np = None
        try:
            import pandas as _pd
        except Exception:  # pragma: no cover - pandas is already a dependency
            _pd = None

        if _np is not None:
            if isinstance(obj, (_np.integer,)):
                return int(obj)
            if isinstance(obj, (_np.floating,)):
                return float(obj)
            if isinstance(obj, _np.ndarray):
                return obj.tolist()

        if _pd is not None:
            if isinstance(obj, _pd.DataFrame):
                return obj.to_dict(orient="records")
            if isinstance(obj, _pd.Series):
                return obj.tolist()

        if isinstance(obj, dict):
            # Ensure keys are strings in JSON
            return {str(k): self._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._json_safe(x) for x in obj]
        return obj

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

        # Calculate coherence score
        # Higher coherence is believed to facilitate human understanding by reducing
        # cognitive effort and improving pattern recognition (Lee et al., 2024).
        coherence_model = CoherenceModel(
            model=self._lda_model,
            texts=self._processed_docs,
            dictionary=self._dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()

        # Calculate perplexity
        # Perplexity is a positive value. Lower perplexity scores generally indicate
        # a better model fit and improved predictive accuracy for unseen documents
        # (Mettler et al., 2025).
        perplexity = self._lda_model.log_perplexity(self._bag_of_words)

        if verbose:
            print(f"\nCoherence Score (c_v): {coherence_score:.4f}")
            print(f"Perplexity: {perplexity:.4f}\n")

        # Add scores to corpus metadata
        if self._corpus is not None:
            self._corpus.metadata["coherence_score"] = float(coherence_score)
            self._corpus.metadata["perplexity"] = float(perplexity)

        documents_copy = []
        documents = self._corpus.documents if self._corpus is not None else []
        if verbose:
            for doc_id, topic in clusters.items():
                print(
                    f"Document ID: {doc_id}, Cluster: {topic[0][0]}, Probability: {topic[0][1]}"
                )

        # Add cluster information to documents metadata
        for doc in tqdm(documents, desc="Adding cluster metadata", disable=len(documents) < 10):
            doc_id = doc.id
            if doc_id in clusters:
                cluster = clusters[doc_id][0][0]
                probability = clusters[doc_id][0][1]
                # Cast to JSON-serializable scalars
                doc.metadata["cluster"] = int(cluster)
                doc.metadata["probability"] = float(probability)
                documents_copy.append(doc)
            else:
                # If the document ID is not found in clusters, keep it unchanged
                documents_copy.append(doc)
        # Update the corpus with the modified documents
        if self._corpus is not None:
            self._corpus.documents = documents_copy
        # Add cluster information to corpus metadata
        if self._corpus is not None:
            # Sanitize clusters to be JSON-safe
            safe_clusters = {}
            for k, v in clusters.items():
                safe_pairs = []
                for pair in v:
                    try:
                        topic_idx, prob = pair
                    except Exception:
                        # Unexpected shape; make a safe conversion
                        topic_idx, prob = pair[0], pair[1] if len(pair) > 1 else 0.0
                    safe_pairs.append([int(topic_idx), float(prob)])
                safe_clusters[str(k)] = safe_pairs
            self._corpus.metadata["clusters"] = safe_clusters
        return clusters

    def format_topics_sentences(self, visualize=False):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        if self._bag_of_words is None:
            raise ValueError(
                "Bag of words is not available. Ensure 'process()' has been called successfully."
            )
        if self._lda_model is None:
            self.build_lda_model()
        if self._lda_model is None:
            raise ValueError("LDA model could not be built.")
        for i, row_list in enumerate(self._lda_model[self._bag_of_words]):
            row = row_list[0] if self._lda_model.per_word_topics else row_list
            # print(row)
            if isinstance(row, list):
                # Ensure all prop_topic values are native Python float
                row = [(topic_num, float(prop_topic)) for topic_num, prop_topic in row]
                row = sorted(row, key=lambda x: (x[1]), reverse=True)
            elif (
                isinstance(row, tuple)
                and len(row) == 2
                and all(isinstance(x, (int, float)) for x in row)
            ):
                # Convert prop_topic to native float if needed
                topic_num, prop_topic = row
                row = [(topic_num, float(prop_topic))]
            else:
                row = []
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = self._lda_model.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    new_row = pd.DataFrame(
                        [
                            [
                                str(self._ids[i]),
                                int(topic_num),
                                float(round(prop_topic, 4)),
                                str(topic_keywords),
                            ]
                        ],
                        columns=[
                            "Title",
                            "Dominant_Topic",
                            "Perc_Contribution",
                            "Topic_Keywords",
                        ],
                    )
                    sent_topics_df = pd.concat(
                        [sent_topics_df, new_row], ignore_index=True
                    )
                else:
                    break
        sent_topics_df.columns = [
            "Title",
            "Dominant_Topic",
            "Perc_Contribution",
            "Topic_Keywords",
        ]

        documents_copy = []
        documents = self._corpus.documents if self._corpus is not None else []
        # Add topic information to the documents metadata
        for doc in tqdm(documents, desc="Adding topic metadata", disable=len(documents) < 10):
            doc_id_str = str(doc.id)
            if doc_id_str in sent_topics_df["Title"].values:
                topic = int(
                    sent_topics_df[sent_topics_df["Title"] == doc_id_str][
                        "Dominant_Topic"
                    ].values[0]
                )
                probability = float(
                    sent_topics_df[sent_topics_df["Title"] == doc_id_str][
                        "Perc_Contribution"
                    ].values[0]
                )
                keywords = str(
                    sent_topics_df[sent_topics_df["Title"] == doc_id_str][
                        "Topic_Keywords"
                    ].values[0]
                )
                doc.metadata["topic"] = topic
                doc.metadata["probability"] = probability
                doc.metadata["keywords"] = keywords
                documents_copy.append(doc)
            else:
                # If the document ID is not found in clusters, keep it unchanged
                documents_copy.append(doc)

        # Add original text to the end of the output
        if visualize:
            contents = pd.Series(self._processed_docs)
            contents.name = "Text"
            sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        # Add to visualize (store as JSON-serializable records)
        self._corpus.visualization["assign_topics"] = sent_topics_df.reset_index(
            drop=False
        ).to_dict(
            orient="records"
        )  # type: ignore
        print("\n Document Topics: \n")
        print(
            tabulate(
                sent_topics_df.head(10),  # type: ignore
                headers="keys",
                tablefmt="psql",
                showindex=False,
                numalign="left",
                stralign="left",
            )
        )
        return sent_topics_df.reset_index(drop=False)

    # https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/
    def most_representative_docs(self):
        sent_topics_df = self.format_topics_sentences()
        sent_topics_sorteddf_mallet = pd.DataFrame()
        sent_topics_outdf_grpd = sent_topics_df.groupby("Dominant_Topic")

        for i, grp in sent_topics_outdf_grpd:
            sent_topics_sorteddf_mallet = pd.concat(
                [
                    sent_topics_sorteddf_mallet,
                    grp.sort_values(["Perc_Contribution"], ascending=False).head(1),
                ],
                axis=0,
            )
        # Add to visualize (JSON-serializable)
        self._corpus.visualization["most_representative_docs"] = (
            sent_topics_sorteddf_mallet.to_dict(orient="records")  # type: ignore
        )
        return sent_topics_sorteddf_mallet

    def topics_per_document(self, start=0, end=1):
        # Get main topic in each document
        if self._bag_of_words is None:
            raise ValueError(
                "Bag of words is not available. Ensure 'process()' has been called successfully."
            )
        if self._lda_model is None:
            self.build_lda_model()
        if self._lda_model is None:
            raise ValueError("LDA model could not be built.")
        corpus_sel = self._bag_of_words[start:end]
        dominant_topics = []
        topic_percentages = []
        for i, corp in enumerate(corpus_sel):
            topic_percs = self._lda_model[corp]
            dominant_topic = sorted(topic_percs, key=lambda x: x[1], reverse=True)[0][0]
            dominant_topics.append((i, dominant_topic))
            topic_percentages.append(topic_percs)
        # Add to corpus metadata (JSON-serializable)
        if self._corpus is not None:
            self._corpus.metadata["dominant_topics"] = [
                [int(idx), int(topic)] for idx, topic in dominant_topics
            ]
            safe_topic_percentages = []
            for doc_topics in topic_percentages:
                safe_doc_topics = []
                for topic_idx, prob in doc_topics:
                    safe_doc_topics.append([int(topic_idx), float(prob)])
                safe_topic_percentages.append(safe_doc_topics)
            self._corpus.metadata["topic_percentages"] = safe_topic_percentages
        return (dominant_topics, topic_percentages)

    def doc_vectorizer(self, doc, model):
        doc_vector = []
        num_words = 0
        for word in doc:
            try:
                if num_words == 0:
                    doc_vector = model.wv[word]
                else:
                    doc_vector = np.add(doc_vector, model.wv[word])
                num_words += 1
            except:
                # pass if word is not found
                pass
        return np.asarray(doc_vector) / num_words

    def vectorizer(self, docs, titles, num_clusters=4, visualize=False):
        X = []
        T = []
        if self._word2vec_model is None:
            self._word2vec_model = Word2Vec(docs, min_count=20, vector_size=50)
        for index, doc in enumerate(docs):
            X.append(self.doc_vectorizer(doc, self._word2vec_model))
            T.append(titles[index])
        print("Averaged text w2v representstion:")
        print(X[0])
        _X = np.array(X)
        print(_X.shape)
        tsne = TSNE(n_components=2, random_state=0)
        tsne_model = tsne.fit_transform(_X)
        # Obtain the prediction
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        y_pred = kmeans.fit(tsne_model).predict(tsne_model)
        data = pd.DataFrame(
            np.concatenate([tsne_model, y_pred[:, None]], axis=1),
            columns=["x", "y", "colour"],
        )
        # Add the titles to the DataFrame
        data["title"] = T
        if not visualize:
            print(
                tabulate(
                    data,  # type: ignore
                    headers="keys",
                    tablefmt="psql",
                    showindex=False,
                    numalign="left",
                    stralign="left",
                )
            )
        # Add to visualization (JSON-serializable)
        self._corpus.visualization["vectorizer"] = data.to_dict(orient="records")  # type: ignore
        return data
