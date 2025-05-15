"""
Copyright (C) 2020 Bell Eapen

This file is part of qrmine.

qrmine is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

qrmine is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with qrmine.  If not, see <http://www.gnu.org/licenses/>.
"""

import spacy
import operator
from textacy import preprocessing
from .model import Document
from typing import Optional


class Text:

    def __init__(self, document: Optional[Document] = None, lang="en_core_web_sm", max_length=1100000):
        self._document = document
        self._lang = lang
        self._max_length = max_length
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
        self.process_tokens()

    @property
    def document(self):
        """
        Get the document.
        """
        if self._document is None:
            raise ValueError("Document is not set")
        return self._document

    @property
    def max_length(self):
        """
        Get the maximum length of the document.
        """
        return self._max_length

    @property
    def lang(self):
        """
        Get the language of the document.
        """
        return self._lang

    @document.setter
    def document(self, document: Document):
        """
        Set the document.
        """
        if not isinstance(document, Document):
            raise ValueError("Document must be of type Document")
        self._document = document
        self._spacy_doc = None  # Reset spacy_doc when a new document is set
        self._lemma = {}
        self._pos = {}
        self._pos_ = {}
        self._word = {}
        self._sentiment = {}
        self._tag = {}
        self._dep = {}
        self._prob = {}
        self._idx = {}
        self.process_tokens()

    @max_length.setter
    def max_length(self, max_length: int):
        """
        Set the maximum length of the document.
        """
        if not isinstance(max_length, int):
            raise ValueError("max_length must be an integer")
        self._max_length = max_length
        if self._spacy_doc is not None:
            self._spacy_doc.max_length = max_length

    @lang.setter
    def lang(self, lang: str):
        """
        Set the language of the document.
        """
        if not isinstance(lang, str):
            raise ValueError("lang must be a string")
        self._lang = lang
        self.process_tokens()

    def make_spacy_doc(self):
        if self._document is None:
            raise ValueError("Document is not set")
        text = self.process_text(self._document.text)
        metadata = self._document.metadata
        nlp = spacy.load(self._lang)
        nlp.max_length = self._max_length
        self._spacy_doc = nlp(text)
        self._spacy_doc.user_data["metadata"] = metadata
        return self._spacy_doc

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

    def process_tokens(self):
        if self._spacy_doc is None:
            spacy_doc = self.make_spacy_doc()
        else:
            spacy_doc = self._spacy_doc
        for token in spacy_doc:
            if token.is_stop or token.is_digit or token.is_punct or token.is_space:
                continue
            if token.like_url or token.like_num or token.like_email:
                continue
            if len(token.text) < 3 or token.text.isupper():
                continue
            self._lemma[token] = token.lemma_
            self._pos[token] = token.pos_
            self._pos_[token] = token.pos
            self._word[token] = token.lemma_
            self._sentiment = token.sentiment
            self._tag = token.tag_
            self._dep = token.dep_
            self._prob = token.prob
            self._idx = token.idx

    def common_words(self, index=10):
        _words = {}
        for key, value in self._word.items():
            _words[value] = _words.get(value, 0) + 1
        return sorted(_words.items(), key=operator.itemgetter(1), reverse=True)[:index]

    def common_nouns(self, index=10):
        _words = {}
        for key, value in self._word.items():
            if self._pos.get(key, None) == "NOUN":
                _words[value] = _words.get(value, 0) + 1
        return sorted(_words.items(), key=operator.itemgetter(1), reverse=True)[:index]

    def common_verbs(self, index=10):
        _words = {}
        for key, value in self._word.items():
            if self._pos.get(key, None) == "VERB":
                _words[value] = _words.get(value, 0) + 1
        return sorted(_words.items(), key=operator.itemgetter(1), reverse=True)[:index]

    def sentences_with_common_nouns(self, index=10):
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
            for token in span:
                if word in self._word.get(token, " "):
                    spans.append(span)
        return spans

    def dimensions(self, word, index=3):
        _spans = self.spans_with_common_nouns(word)
        _ad = {}
        for span in _spans:
            for token in span:
                if self._pos.get(token, None) == "ADJ":
                    _ad[self._word.get(token)] = _ad.get(self._word.get(token), 0) + 1
                if self._pos.get(token, None) == "ADV":
                    _ad[self._word.get(token)] = _ad.get(self._word.get(token), 0) + 1
                if self._pos.get(token, None) == "VERB":
                    _ad[self._word.get(token)] = _ad.get(self._word.get(token), 0) + 1
        return sorted(_ad.items(), key=operator.itemgetter(1), reverse=True)[:index]

    def attributes(self, word, index=3):
        _spans = self.spans_with_common_nouns(word)
        _ad = {}
        for span in _spans:
            for token in span:
                if self._pos.get(token, None) == "NOUN" and word not in self._word.get(
                    token, ""
                ):
                    _ad[self._word.get(token)] = _ad.get(self._word.get(token), 0) + 1
                    # if self._pos.get(token, None) == 'VERB':
                    # _ad[self._word.get(token)] = _ad.get(self._word.get(token), 0) + 1
        return sorted(_ad.items(), key=operator.itemgetter(1), reverse=True)[:index]

    def generate_summary(self, weight=10):
        """[summary]

        Args:
            weight (int, optional): Parameter for summary generation weight. Defaults to 10.

        Returns:
            list: A list of summary lines
        """
        words = self.common_words()
        spans = []
        ct = 0
        for key, value in words:
            ct += 1
            if ct > weight:
                continue
            for span in self.spans_with_common_nouns(key):
                spans.append(span.text)
        return list(dict.fromkeys(spans))  # remove duplicates