"""A simple alternative tokenizer which parses text by splitting on whitespace."""
from builtins import object

import numpy as np

from fonduer.parser.models.utils import construct_stable_id


class SimpleTokenizer(object):
    """Tokenizes text on whitespace only using split()."""

    def __init__(self, delim="<NB>"):
        self.delim = delim

    def parse(self, document, contents):
        """Parse the document.

        :param document: The Document context of the data model.
        :param contents: The text contents of the document.
        :rtype: a *generator* of tokenized text.
        """
        i = 0
        for text in contents.split(self.delim):
            if not len(text.strip()):
                continue
            words = text.split()
            char_offsets = [0] + [
                int(_) for _ in np.cumsum([len(x) + 1 for x in words])[:-1]
            ]
            text = " ".join(words)
            stable_id = construct_stable_id(document, "sentence", i, i)
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
                "stable_id": stable_id,
            }
            i += 1
