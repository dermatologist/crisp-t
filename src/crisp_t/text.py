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
import multiprocessing
import operator
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import textacy
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from spacy.tokens import Doc
from textacy import preprocessing
from tqdm import tqdm

from .model import Corpus, SpacyManager
from .utils import QRUtils

warnings.filterwarnings("ignore", category=DeprecationWarning)

textacy.set_doc_extensions("extract.bags")  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Text:

    def __init__(
        self, corpus: Corpus | None = None, lang="en_core_web_sm", max_length=1100000
    ):
        self._corpus = corpus
        self._lang = lang
        self._spacy_manager = SpacyManager(self._lang)
        self._max_length = max_length
        self._initial_document_count = len(self._corpus.documents) if corpus else 0  # type: ignore

        self._spacy_doc = None
        self._lemma = {}
        self._pos = {}
        self._pos_ = {}
        self._word = {}
        self._sentiment = {}
        self._tag = {}
        self._dep = {}
        self._prob = {}
        self._idx = {}

    @property
    def corpus(self):
        """
        Get the corpus.
        """
        if self._corpus is None:
            raise ValueError("Corpus is not set")
        return self._corpus

    @property
    def max_length(self):
        """
        Get the maximum length of the corpus.
        """
        return self._max_length

    @property
    def lang(self):
        """
        Get the language of the corpus.
        """
        return self._lang

    @property
    def initial_document_count(self):
        """
        Get the initial document count.
        """
        return self._initial_document_count

    @corpus.setter
    def corpus(self, corpus: Corpus):
        """
        Set the corpus.
        """
        if not isinstance(corpus, Corpus):
            raise ValueError("Corpus must be of type Corpus")
        self._corpus = corpus
        spacy_doc, results = self.process_tokens(self._corpus.id if self._corpus else None)
        self._spacy_doc = spacy_doc
        self._lemma = results["lemma"]
        self._pos = results["pos"]
        self._pos_ = results["pos_"]
        self._word = results["word"]
        self._sentiment = results["sentiment"]
        self._tag = results["tag"]
        self._dep = results["dep"]
        self._prob = results["prob"]
        self._idx = results["idx"]

    @max_length.setter
    def max_length(self, max_length: int):
        """
        Set the maximum length of the corpus.
        """
        if not isinstance(max_length, int):
            raise ValueError("max_length must be an integer")
        self._max_length = max_length
        if self._spacy_doc is not None:
            self._spacy_doc.max_length = max_length

    @lang.setter
    def lang(self, lang: str):
        """
        Set the language of the corpus.
        """
        if not isinstance(lang, str):
            raise ValueError("lang must be a string")
        self._lang = lang
        spacy_doc, results = self.process_tokens(self._corpus.id if self._corpus else None)
        self._spacy_doc = spacy_doc
        self._lemma = results["lemma"]
        self._pos = results["pos"]
        self._pos_ = results["pos_"]
        self._word = results["word"]
        self._sentiment = results["sentiment"]
        self._tag = results["tag"]
        self._dep = results["dep"]
        self._prob = results["prob"]
        self._idx = results["idx"]

    def make_spacy_doc(self):
        if self._corpus is None:
            raise ValueError("Corpus is not set")
        # Use list and join for efficient string concatenation instead of +=
        text_parts = []
        for document in tqdm(
            self._corpus.documents,
            desc="Processing documents",
            disable=len(self._corpus.documents) < 10,
        ):
            text_parts.append(self.process_text(document.text))
        text = " \n".join(text_parts)
        nlp = self._spacy_manager.get_model()
        nlp.max_length = self._max_length
        if len(text) > self._max_length:
            logger.warning(
                f"Text length {len(text)} exceeds max_length {self._max_length}."
            )
            text_chunks = [
                text[i : i + self._max_length]
                for i in range(0, len(text), self._max_length)
            ]
            spacy_docs = []
            for chunk in tqdm(
                text_chunks, desc="Processing text as chunks of max_length"
            ):
                spacy_doc = nlp(chunk)
                spacy_docs.append(spacy_doc)
            self._spacy_doc = spacy_docs[0]
            for doc in tqdm(spacy_docs[1:], desc="Merging spacy docs"):
                self._spacy_doc = Doc.from_docs([self._spacy_doc, doc])  # type: ignore
        else:
            self._spacy_doc = nlp(text)
        return self._spacy_doc

    # @lru_cache(maxsize=3)
    def make_each_document_into_spacy_doc(self, id="corpus"):
        if self._corpus is None:
            raise ValueError("Corpus is not set")

        # ! if cached file exists, load it
        cache_dir = Path("cache")
        cache_file = cache_dir / f"spacy_docs_{id}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                spacy_docs, ids = pickle.load(f)
            # logger.info("Loaded cached spacy docs and ids.")
            return spacy_docs, ids

        spacy_docs = []
        ids = []
        # Load SpaCy model once outside the loop for efficiency
        nlp = self._spacy_manager.get_model()
        nlp.max_length = self._max_length
        for document in tqdm(
            self._corpus.documents,
            desc="Creating spacy docs",
            disable=len(self._corpus.documents) < 10,
        ):
            text = self.process_text(document.text)
            spacy_doc = nlp(text)
            spacy_docs.append(spacy_doc)
            ids.append(document.id)

        # ! dump spacy_docs, ids to a file for caching with the corpus id
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"spacy_docs_{id}.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump((spacy_docs, ids), f)
        return spacy_docs, ids

    def process_text(self, text: str) -> str:
        """
        Process the text by removing unwanted characters and normalizing it.
        """
        # Remove unwanted characters
        text = preprocessing.replace.urls(text)
        text = preprocessing.replace.emails(text)
        text = preprocessing.replace.phone_numbers(text)
        text = preprocessing.replace.currency_symbols(text)
        text = preprocessing.replace.hashtags(text)
        text = preprocessing.replace.numbers(text)

        # lowercase the text
        text = text.lower()
        return text

    # @lru_cache(maxsize=3)
    def process_tokens(self, id="corpus"):
        """
        Process tokens in the spacy document and extract relevant information.
        """

        # ! if cached file exists, load it
        cache_dir = Path("cache")
        cache_file = cache_dir / f"spacy_doc_{id}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                spacy_doc, results = pickle.load(f)
            # logger.info("Loaded cached spacy doc and results.")
            return spacy_doc, results

        spacy_doc = self.make_spacy_doc()
        logger.info("Spacy doc created.")

        n_cores = multiprocessing.cpu_count()

        def process_token(token):
            if token.is_stop or token.is_digit or token.is_punct or token.is_space:
                return None
            if token.like_url or token.like_num or token.like_email:
                return None
            if len(token.text) < 3 or token.text.isupper():
                return None
            return {
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "pos_": token.pos,
                "word": token.lemma_,
                "sentiment": token.sentiment,
                "tag": token.tag_,
                "dep": token.dep_,
                "prob": token.prob,
                "idx": token.idx,
            }

        tokens = list(spacy_doc)
        _lemma = {}
        _pos = {}
        _pos_ = {}
        _word = {}
        _sentiment = {}
        _tag = {}
        _dep = {}
        _prob = {}
        _idx = {}
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_token, token): token for token in tokens}
            with tqdm(
                total=len(futures),
                desc=f"Processing tokens (parallel, {n_cores} cores)",
            ) as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        _lemma[result["text"]] = result["lemma"]
                        _pos[result["text"]] = result["pos"]
                        _pos_[result["text"]] = result["pos_"]
                        _word[result["text"]] = result["word"]
                        _sentiment[result["text"]] = result["sentiment"]
                        _tag = result["tag"]
                        _dep = result["dep"]
                        _prob = result["prob"]
                        _idx = result["idx"]
                    pbar.update(1)
        logger.info("Token processing complete.")
        results = {
            "lemma": _lemma,
            "pos": _pos,
            "pos_": _pos_,
            "word": _word,
            "sentiment": _sentiment,
            "tag": _tag,
            "dep": _dep,
            "prob": _prob,
            "idx": _idx,
        }
        # ! dump spacy_doc, results to a file for caching with the corpus id
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"spacy_doc_{id}.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump((spacy_doc, results), f)

        return spacy_doc, results

    def map_spacy_doc(self):
        spacy_doc, results = self.process_tokens(self._corpus.id if self._corpus else None)
        self._spacy_doc = spacy_doc
        self._lemma = results["lemma"]
        self._pos = results["pos"]
        self._pos_ = results["pos_"]
        self._word = results["word"]
        self._sentiment = results["sentiment"]
        self._tag = results["tag"]
        self._dep = results["dep"]
        self._prob = results["prob"]
        self._idx = results["idx"]

    def common_words(self, index=10):
        self.map_spacy_doc()
        _words = {}
        for key, value in self._word.items():
            _words[value] = _words.get(value, 0) + 1
        return sorted(_words.items(), key=operator.itemgetter(1), reverse=True)[:index]

    def common_nouns(self, index=10):
        self.map_spacy_doc()
        _words = {}
        for key, value in self._word.items():
            if self._pos.get(key, None) == "NOUN":
                _words[value] = _words.get(value, 0) + 1
        return sorted(_words.items(), key=operator.itemgetter(1), reverse=True)[:index]

    def common_verbs(self, index=10):
        self.map_spacy_doc()
        _words = {}
        for key, value in self._word.items():
            if self._pos.get(key, None) == "VERB":
                _words[value] = _words.get(value, 0) + 1
        return sorted(_words.items(), key=operator.itemgetter(1), reverse=True)[:index]

    def print_coding_dictionary(self, num=10, top_n=5):
        """Prints a coding dictionary based on common verbs, attributes, and dimensions.
        "CATEGORY" is the common verb
        "PROPERTY" is the common nouns associated with the verb
        "DIMENSION" is the common adjectives/adverbs/verbs associated with the property
        Args:
            num (int, optional): Number of common verbs to consider. Defaults to 10.
            top_n (int, optional): Number of top attributes and dimensions to consider for each verb. Defaults to 5.

        """
        self.map_spacy_doc()
        output = []
        coding_dict = []
        output.append(("CATEGORY", "PROPERTY", "DIMENSION"))
        verbs = self.common_verbs(num)
        _verbs = []
        for verb, freq in verbs:
            _verbs.append(verb)
        for verb, freq in verbs:
            for attribute, f2 in self.attributes(verb, top_n):
                for dimension, f3 in self.dimensions(attribute, top_n):
                    if dimension not in _verbs:
                        output.append((verb, attribute, dimension))
                        coding_dict.append(f"{verb} > {attribute} > {dimension}")
        # Add coding_dict to corpus metadata
        if self._corpus is not None:
            self._corpus.metadata["coding_dict"] = coding_dict
        print("\n---Coding Dictionary---")
        QRUtils.print_table(output)
        print("---------------------------\n")
        return output

    def sentences_with_common_nouns(self, index=10):
        self.map_spacy_doc()
        _nouns = self.common_nouns(index)
        # Let's look at the sentences
        sents = []
        # Ensure self._spacy_doc is initialized
        if self._spacy_doc is None:
            self._spacy_doc = self.make_spacy_doc()
        # the "sents" property returns spans
        # spans have indices into the original string
        # where each index value represents a token
        for span in self._spacy_doc.sents:
            # go from the start to the end of each span, returning each token in the sentence
            # combine each token using join()
            sent = " ".join(
                self._spacy_doc[i].text for i in range(span.start, span.end)
            ).strip()
            for noun, freq in _nouns:
                if noun in sent:
                    sents.append(sent)
        return sents

    def spans_with_common_nouns(self, word):
        self.map_spacy_doc()
        # Let's look at the sentences
        spans = []
        # the "sents" property returns spans
        # spans have indices into the original string
        # where each index value represents a token
        if self._spacy_doc is None:
            self._spacy_doc = self.make_spacy_doc()
        for span in self._spacy_doc.sents:
            # go from the start to the end of each span, returning each token in the sentence
            # combine each token using join()
            for token in span.text.split():
                if word in self._word.get(token, " "):
                    spans.append(span)
        return spans

    def dimensions(self, word, index=3):
        self.map_spacy_doc()
        _spans = self.spans_with_common_nouns(word)
        _ad = {}
        for span in _spans:
            for token in span.text.split():
                if self._pos.get(token, None) == "ADJ":
                    _ad[self._word.get(token)] = _ad.get(self._word.get(token), 0) + 1
                if self._pos.get(token, None) == "ADV":
                    _ad[self._word.get(token)] = _ad.get(self._word.get(token), 0) + 1
                if self._pos.get(token, None) == "VERB":
                    _ad[self._word.get(token)] = _ad.get(self._word.get(token), 0) + 1
        return sorted(_ad.items(), key=operator.itemgetter(1), reverse=True)[:index]

    def attributes(self, word, index=3):
        self.map_spacy_doc()
        _spans = self.spans_with_common_nouns(word)
        _ad = {}
        for span in _spans:
            for token in span.text.split():
                if self._pos.get(token, None) == "NOUN" and word not in self._word.get(
                    token, ""
                ):
                    _ad[self._word.get(token)] = _ad.get(self._word.get(token), 0) + 1
                    # if self._pos.get(token, None) == 'VERB':
                    # _ad[self._word.get(token)] = _ad.get(self._word.get(token), 0) + 1
        return sorted(_ad.items(), key=operator.itemgetter(1), reverse=True)[:index]

    # filter documents in the corpus based on metadata
    def filter_documents(self, metadata_key, metadata_value, mcp=False, id_column="id"):
        """
        Filter documents in the corpus based on metadata.
        If id_column exists in self._corpus.df, filter the DataFrame to match filtered documents' ids.
        """
        # * filter does not require spacy mapping
        # self.map_spacy_doc()
        if self._corpus is None:
            raise ValueError("Corpus is not set")
        filtered_documents = []
        for document in tqdm(
            self._corpus.documents,
            desc="Filtering documents",
            disable=len(self._corpus.documents) < 10,
        ):
            meta_val = document.metadata.get(metadata_key)
            # Check meta_val is not None and is iterable (str, list, tuple, set)
            if meta_val is not None and isinstance(meta_val, (str, list, tuple, set)) and metadata_value in meta_val:
                filtered_documents.append(document)
            # Check document.id and document.text are not None and are str
            if isinstance(document.id, str) and metadata_value in document.id:
                filtered_documents.append(document)
            if isinstance(document.name, str) and metadata_value in document.name:
                filtered_documents.append(document)
        self._corpus.documents = filtered_documents

        # Check for id_column in self._corpus.df and filter df if present
        if (
            hasattr(self._corpus, "df")
            and self._corpus.df is not None
            and id_column in self._corpus.df.columns
        ):
            logger.info(f"id_column '{id_column}' exists in DataFrame.")
            filtered_ids = [doc.id for doc in filtered_documents]
            # Convert id_column to string before comparison
            self._corpus.df = self._corpus.df[
                self._corpus.df[id_column]
                .astype(str)
                .isin([str(i) for i in filtered_ids])
            ]
        else:
            logger.warning(f"id_column '{id_column}' does not exist in DataFrame.")

        if mcp:
            return f"Filtered {len(filtered_documents)} documents with {metadata_key} containing {metadata_value}"
        return filtered_documents

    # get the count of documents in the corpus
    def document_count(self):
        """
        Get the count of documents in the corpus.
        """
        if self._corpus is None:
            raise ValueError("Corpus is not set")
        return len(self._corpus.documents)

    def generate_summary(self, weight=10):
        """[summary]

        Args:
            weight (int, optional): Parameter for summary generation weight. Defaults to 10.

        Returns:
            list: A list of summary lines
        """
        self.map_spacy_doc()
        words = self.common_words()
        spans = []
        ct = 0
        for key, value in words:
            ct += 1
            if ct > weight:
                continue
            for span in self.spans_with_common_nouns(key):
                spans.append(span.text)
        if self._corpus is not None:
            self._corpus.metadata["summary"] = list(
                dict.fromkeys(spans)
            )  # remove duplicates
        return list(dict.fromkeys(spans))  # remove duplicates

    def print_categories(self, spacy_doc=None, num=10):
        self.map_spacy_doc()
        bot = self._spacy_doc._.to_bag_of_terms( # type: ignore
            by="lemma_",
            weighting="freq",
            ngs=(1, 2, 3),
            ents=True,
            ncs=True,
            dedupe=True,
        )
        categories = sorted(bot.items(), key=lambda x: x[1], reverse=True)[:num]
        output = []
        to_return = []
        print("\n---Categories with count---")
        output.append(("CATEGORY", "WEIGHT"))
        for category, count in categories:
            output.append((category, str(count)))
            to_return.append(category)
        QRUtils.print_table(output)
        print("---------------------------\n")
        if self._corpus is not None:
            self._corpus.metadata["categories"] = output
        return to_return

    def category_basket(self, num=10):
        item_basket = []
        spacy_docs, ids = self.make_each_document_into_spacy_doc()
        for spacy_doc in spacy_docs:
            item_basket.append(self.print_categories(spacy_doc, num))
        documents_copy = []
        documents = self._corpus.documents if self._corpus is not None else []
        # add cateogies to respective documents
        for i, document in enumerate(documents):
            if i < len(item_basket):
                document.metadata["categories"] = item_basket[i]
                documents_copy.append(document)
        # update the corpus with the new documents
        if self._corpus is not None:
            self._corpus.documents = documents_copy
        return item_basket
        # Example return:
        # [['GT', 'Strauss', 'coding', 'ground', 'theory', 'seminal', 'Corbin', 'code',
        # 'structure', 'ground theory'], ['category', 'theory', 'comparison', 'incident',
        # 'GT', 'structure', 'coding', 'Classical', 'Grounded', 'Theory'],
        # ['theory', 'GT', 'evaluation'], ['open', 'coding', 'category', 'QRMine',
        # 'open coding', 'researcher', 'step', 'data', 'break', 'analytically'],
        # ['ground', 'theory', 'GT', 'ground theory'], ['category', 'comparison', 'incident',
        # 'category comparison', 'Theory', 'theory']]

    def category_association(self, num=10):
        """Generates the support for itemsets

        Args:
            num (int, optional): number of categories to generate for each doc in corpus. . Defaults to 10.
        """
        self.map_spacy_doc()
        basket = self.category_basket(num)
        te = TransactionEncoder()
        te_ary = te.fit(basket).transform(basket)
        df = pd.DataFrame(te_ary, columns=te.columns_)  # type: ignore
        _apriori = apriori(df, min_support=0.6, use_colnames=True)
        # Example
        #    support      itemsets
        # 0  0.666667          (GT)
        # 1  0.833333      (theory)
        # 2  0.666667  (theory, GT)
        documents_copy = []
        documents = self._corpus.documents if self._corpus is not None else []
        # TODO (Change) Add association rules to each document
        for i, document in enumerate(documents):
            if i < len(basket):
                # ! fix document.metadata["association_rules"] = _apriori #TODO This is a corpus metadata, not a document one
                documents_copy.append(document)
        # Add to corpus metadata
        if self._corpus is not None:
            self._corpus.metadata["association_rules"] = _apriori
        # Update the corpus with the new documents
        if self._corpus is not None:
            self._corpus.documents = documents_copy
        return _apriori
