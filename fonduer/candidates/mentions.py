import logging
import re
from builtins import range

from fonduer.candidates.models import TemporaryImage, TemporarySpan
from fonduer.parser.models import Document

logger = logging.getLogger(__name__)


class MentionSpace(object):
    """Defines the **space** of Mention objects.

    Calling _apply(x)_ given an object _x_ returns a generator over mentions in _x_.
    """

    def __init__(self):
        pass

    def apply(self, x):
        raise NotImplementedError()


class Ngrams(MentionSpace):
    """
    Defines the space of mentions as all n-grams (n <= n_max) in a Sentence _x_,
    indexing by **character offset**.
    """

    def __init__(self, n_max=5, split_tokens=("-", "/")):
        MentionSpace.__init__(self)
        self.n_max = n_max
        self.split_rgx = (
            r"(" + r"|".join(split_tokens) + r")"
            if split_tokens and len(split_tokens) > 0
            else None
        )

    def apply(self, context):

        # These are the character offset--**relative to the sentence
        # start**--for each _token_
        offsets = context.char_offsets

        # Loop over all n-grams in **reverse** order (to facilitate
        # longest-match semantics)
        L = len(offsets)
        seen = set()
        for j in range(1, self.n_max + 1)[::-1]:
            for i in range(L - j + 1):
                w = context.words[i + j - 1]
                start = offsets[i]
                end = offsets[i + j - 1] + len(w) - 1
                ts = TemporarySpan(char_start=start, char_end=end, sentence=context)
                if ts not in seen:
                    seen.add(ts)
                    yield ts

                # Check for split
                # NOTE: For simplicity, we only split single tokens right now!
                if j == 1 and self.split_rgx is not None and end - start > 0:
                    m = re.search(
                        self.split_rgx,
                        context.text[start - offsets[0] : end - offsets[0] + 1],
                    )
                    if m is not None and j < self.n_max + 1:
                        ts1 = TemporarySpan(
                            char_start=start,
                            char_end=start + m.start(1) - 1,
                            sentence=context,
                        )
                        if ts1 not in seen:
                            seen.add(ts1)
                            yield ts
                        ts2 = TemporarySpan(
                            char_start=start + m.end(1), char_end=end, sentence=context
                        )
                        if ts2 not in seen:
                            seen.add(ts2)
                            yield ts2


class MentionNgrams(Ngrams):
    """Defines the **space** of Mentions.

    Defines the space of mentions as all n-grams (n <= n_max) in a Document _x_,
    divided into Sentences inside of html elements (such as table cells).
    """

    def __init__(self, n_max=5, split_tokens=["-", "/"]):
        """
        Initialize MentionNgrams.
        """
        Ngrams.__init__(self, n_max=n_max, split_tokens=split_tokens)

    def apply(self, session, context):
        """
        Generate MentionNgrams from a Document by parsing all of its Sentences.
        """
        if not isinstance(context, Document):
            raise TypeError(
                "Input Contexts to MentionNgrams.apply() must be of type Document"
            )

        doc = session.query(Document).filter(Document.id == context.id).one()
        for sentence in doc.sentences:
            for ts in Ngrams.apply(self, sentence):
                yield ts


class MentionFigures(MentionSpace):
    """
    Defines the space of mentions as all figures in a Document _x_,
    indexing by **position offset**.
    """

    def __init__(self, type=None):
        """
        Initialize MentionFigures.

        Only support figure type filter.
        """
        MentionSpace.__init__(self)
        if type is not None:
            self.type = type.strip().lower()
        self.type = None

    def apply(self, session, context):
        """
        Generate MentionFigures from a Document by parsing all of its Figures.
        """
        if not isinstance(context, Document):
            raise TypeError(
                "Input Contexts to MentionFigures.apply() must be of type Document"
            )

        doc = session.query(Document).filter(Document.id == context.id).one()
        for figure in doc.figures:
            if self.type is None or figure.url.lower().endswith(self.type):
                yield TemporaryImage(figure)
