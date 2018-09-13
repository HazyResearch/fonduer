import logging
import re
from builtins import map, range
from copy import deepcopy

from sqlalchemy.sql import select

from fonduer.candidates.models import Mention, TemporaryImage, TemporarySpan
from fonduer.parser.models import Document
from fonduer.utils.udf import UDF, UDFRunner

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
    Defines the space of Mentions as all n-grams (n_min <= n <= n_max) in a Sentence
    _x_, indexing by **character offset**.

    :param n_min: Lower limit for the generated n_grams.
    :param n_max: Upper limit for the generated n_grams.
    :param split_tokens: Tokens, on which unigrams are split into two separate unigrams.

    """

    def __init__(self, n_min=1, n_max=5, split_tokens=("-", "/")):
        MentionSpace.__init__(self)
        self.n_min = n_min
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
        for j in range(self.n_min, self.n_max + 1)[::-1]:
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
                if (
                    j == 1
                    and self.n_max >= 1
                    and self.n_min <= 1
                    and self.split_rgx is not None
                    and end - start > 0
                ):
                    m = re.search(
                        self.split_rgx,
                        context.text[start - offsets[0] : end - offsets[0] + 1],
                    )
                    if m is not None:
                        ts1 = TemporarySpan(
                            char_start=start,
                            char_end=start + m.start(1) - 1,
                            sentence=context,
                        )
                        if ts1 not in seen and ts1.get_span():
                            seen.add(ts1)
                            yield ts1
                        ts2 = TemporarySpan(
                            char_start=start + m.end(1), char_end=end, sentence=context
                        )
                        if ts2 not in seen and ts2.get_span():
                            seen.add(ts2)
                            yield ts2


class MentionNgrams(Ngrams):
    """Defines the **space** of Mentions.

    Defines the space of Mentions as all n-grams (n_min <= n <= n_max) in a Document
    _x_, divided into Sentences inside of html elements (such as table cells).

    :param n_min: Lower limit for the generated n_grams.
    :param n_max: Upper limit for the generated n_grams.
    :param split_tokens: Tokens, on which unigrams are split into two separate unigrams.
    """

    def __init__(self, n_min=1, n_max=5, split_tokens=["-", "/"]):
        """
        Initialize MentionNgrams.
        """
        Ngrams.__init__(self, n_min=n_min, n_max=n_max, split_tokens=split_tokens)

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
    Defines the space of Mentions as all figures in a Document _x_,
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


class MentionExtractor(UDFRunner):
    """An operator to extract Mention objects from a Context.

    :param session: An initialized database session.
    :param mention_classes: The type of relation to extract, defined using
        :func: fonduer.mentions.mention_subclass.
    :param mention_spaces: one or list of :class:`MentionSpace` objects, one for
        each relation argument. Defines space of Contexts to consider
    :param matchers: one or list of :class:`fonduer.matchers.Matcher` objects,
        one for each relation argument. Only tuples of Contexts for which each
        element is accepted by the corresponding Matcher will be returned as
        Mentions
    """

    def __init__(
        self, session, mention_classes, mention_spaces, matchers, parallelism=1
    ):
        """Initialize the MentionExtractor."""
        super(MentionExtractor, self).__init__(
            session,
            MentionExtractorUDF,
            parallelism=parallelism,
            mention_classes=mention_classes,
            mention_spaces=mention_spaces,
            matchers=matchers,
        )
        # Check that arity is same
        arity = len(mention_classes)
        if not all(
            len(x) == arity for x in [mention_classes, mention_spaces, matchers]
        ):
            raise ValueError(
                "Mismatched arity of mention classes, spaces, and matchers."
            )

        self.mention_classes = mention_classes

    def apply(self, xs, split=0, **kwargs):
        """Call the MentionExtractorUDF."""
        super(MentionExtractor, self).apply(xs, split=split, **kwargs)

    def clear(self, **kwargs):
        """Delete Mentions of each class in the extractor from the given split."""
        for mention_class in self.mention_classes:
            logger.info("Clearing table: {}".format(mention_class.__tablename__))
            self.session.query(Mention).filter(
                Mention.type == mention_class.__tablename__
            ).delete()

    def clear_all(self, **kwargs):
        """Delete all Mentions from given split the database."""
        logger.info("Clearing ALL Mentions.")
        self.session.query(Mention).delete()

    def get_mentions(self, docs=None):
        """Return a list of lists of the mentions associated with this extractor.

        Each list of the return will contain the Mentions for one of the
        mention classes associated with the MentionExtractor.

        :param docs: If provided, return Mentions from these documents. Else,
            return all Mentions.
        :return: List of lists of Mentions for each mention_class.
        """
        result = []
        if docs:
            docs = docs if isinstance(docs, (list, tuple)) else [docs]
            # Get cands from all splits
            for mention_class in self.mention_classes:
                mentions = (
                    self.session.query(mention_class)
                    .filter(mention_class.document_id.in_([doc.id for doc in docs]))
                    .order_by(mention_class.id)
                    .all()
                )
                result.append(mentions)
        else:
            for mention_class in self.mention_classes:
                mentions = (
                    self.session.query(mention_class).order_by(mention_class.id).all()
                )
                result.append(mentions)
        return result


class MentionExtractorUDF(UDF):
    """UDF for performing mention extraction."""

    def __init__(self, mention_classes, mention_spaces, matchers, **kwargs):
        """Initialize the MentionExtractorUDF."""
        self.mention_classes = (
            mention_classes
            if isinstance(mention_classes, (list, tuple))
            else [mention_classes]
        )
        self.mention_spaces = (
            mention_spaces
            if isinstance(mention_spaces, (list, tuple))
            else [mention_spaces]
        )
        self.matchers = matchers if isinstance(matchers, (list, tuple)) else [matchers]

        # Make sure the mention spaces are different so generators aren't expended!
        self.mention_spaces = list(map(deepcopy, self.mention_spaces))

        # Preallocates internal data structure
        self.child_context_set = set()

        super(MentionExtractorUDF, self).__init__(**kwargs)

    def apply(self, context, clear, **kwargs):
        """Extract mentions from the given Context.

        :param context: A document to process.
        :param clear: Whether or not to clear the existing database entries.
        """

        # Iterate over each mention class
        for i, mention_class in enumerate(self.mention_classes):
            # Generate TemporaryContexts that are children of the context using the
            # mention_space and filtered by the Matcher
            self.child_context_set.clear()
            for tc in self.matchers[i].apply(
                self.mention_spaces[i].apply(self.session, context)
            ):
                tc._load_id_or_insert(self.session)
                self.child_context_set.add(tc)

            # Generates and persists mentions
            mention_args = {"document_id": context.id}
            for child_context in self.child_context_set:
                # Assemble mention arguments
                for arg_name in mention_class.__argnames__:
                    mention_args[arg_name + "_id"] = child_context.id

                # Checking for existence
                if not clear:
                    q = select([mention_class.id])
                    for key, value in list(mention_args.items()):
                        q = q.where(getattr(mention_class, key) == value)
                    mention_id = self.session.execute(q).first()
                    if mention_id is not None:
                        continue

                # Add Mention to session
                yield mention_class(**mention_args)
