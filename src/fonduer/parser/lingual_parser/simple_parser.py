"""A simple alternative tokenizer which parses text by splitting on whitespace."""
from typing import Any, Dict, Iterator

import numpy as np

from fonduer.parser.lingual_parser.lingual_parser import LingualParser


class SimpleParser(LingualParser):
    """Tokenizes text on whitespace only using split().

    :param delim: a delimiter to split text into sentences.
    """

    def __init__(self, delim: str = ".") -> None:
        """Initialize SimpleParser."""
        self.delim = delim

    def split_sentences(self, str: str) -> Iterator[Dict[str, Any]]:
        """Parse the document.

        :param str: The text contents of the document.
        :return: a *generator* of tokenized text.
        """
        i = 0
        for text in str.split(self.delim):
            if not len(text.strip()):
                continue
            words = text.split()
            char_offsets = [0] + [
                int(_) for _ in np.cumsum([len(x) + 1 for x in words])[:-1]
            ]
            text = " ".join(words)
            yield {
                "text": text,
                "words": words,
                "pos_tags": [""] * len(words),
                "ner_tags": [""] * len(words),
                "lemmas": [""] * len(words),
                "dep_parents": [0] * len(words),
                "dep_labels": [""] * len(words),
                "char_offsets": char_offsets,
                "abs_char_offsets": char_offsets,
            }
            i += 1

    def has_NLP_support(self) -> bool:
        """
        Return True when NLP is supported.

        :return: True when NLP is supported.
        """
        return False

    def has_tokenizer_support(self) -> bool:
        """
        Return True when a tokenizer is supported.

        :return: True when a tokenizer is supported.
        """
        return True
