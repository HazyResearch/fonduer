"""Fonduer lingual parser."""
from typing import Collection, Iterable, Iterator

from fonduer.parser.models import Sentence


class LingualParser(object):
    """Lingual parser."""

    def split_sentences(self, text: str) -> Iterable[dict]:
        """
        Split input text into sentences.

        :param text: text to be split
        :return: A generator of dict that is used as `**kwargs` to instantiate
            :class:`Sentence <fonduer.parser.models.Sentence>`.
        """
        raise NotImplementedError()

    def enrich_sentences_with_NLP(
        self, sentences: Collection[Sentence]
    ) -> Iterator[Sentence]:
        """
        Add NLP attributes like lemmas, pos_tags, etc. to sentences.

        :param sentences: a iterator of
            :class:`Sentence <fonduer.parser.models.Sentence>`.
        :return: a generator of :class:`Sentence <fonduer.parser.models.Sentence>`.
        """
        raise NotImplementedError()

    def has_NLP_support(self) -> bool:
        """
        Return True when NLP is supported.

        :return: True when NLP is supported.
        """
        raise NotImplementedError()

    def has_tokenizer_support(self) -> bool:
        """
        Return True when a tokenizer is supported.

        :return: True when a tokenizer is supported.
        """
        raise NotImplementedError()
