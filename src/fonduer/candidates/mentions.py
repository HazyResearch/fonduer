import logging
import re
from builtins import map, range
from collections import defaultdict
from copy import deepcopy

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.sql import select

from fonduer.candidates.models import Mention
from fonduer.candidates.models.image import TemporaryImage
from fonduer.candidates.models.span import TemporarySpan
from fonduer.parser.models import Document
from fonduer.utils.udf import UDF, UDFRunner

logger = logging.getLogger(__name__)


class MentionSpace(object):
    """Defines the **space** of Mention objects.

    Calling *apply(x)* given an object *x* returns a generator over mentions in
    *x*.
    """

    def __init__(self):
        pass

    def apply(self, x):
        raise NotImplementedError()


class Ngrams(MentionSpace):
    """
    Defines the space of Mentions as all n-grams (n_min <= n <= n_max) in a
    Sentence *x*, indexing by **character offset**.

    :param n_min: Lower limit for the generated n_grams.
    :type n_min: int
    :param n_max: Upper limit for the generated n_grams.
    :type n_max: int
    :param split_tokens: Tokens, on which unigrams are split into two separate
        unigrams.
    :type split_tokens: tuple, list of str.
    """

    def __init__(self, n_min=1, n_max=5, split_tokens=[]):
        MentionSpace.__init__(self)
        self.n_min = n_min
        self.n_max = n_max
        self.split_rgx = (
            r"(" + r"|".join(map(re.escape, sorted(split_tokens, reverse=True))) + r")"
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
                if (
                    j == 1
                    and self.n_max >= 1
                    and self.n_min <= 1
                    and self.split_rgx is not None
                    and end - start > 0
                ):
                    text = context.text[start - offsets[0] : end - offsets[0] + 1]
                    start_idxs = [0]
                    end_idxs = []
                    for m in re.finditer(self.split_rgx, text):
                        start_idxs.append(m.end())
                        end_idxs.append(m.start())
                    end_idxs.append(len(text))
                    for start_idx in start_idxs:
                        for end_idx in end_idxs:
                            if start_idx < end_idx:
                                ts = TemporarySpan(
                                    char_start=start_idx,
                                    char_end=end_idx - 1,
                                    sentence=context,
                                )
                                if ts not in seen and ts.get_span():
                                    seen.add(ts)
                                    yield ts


class MentionNgrams(Ngrams):
    """Defines the **space** of Mentions.

    Defines the space of Mentions as all n-grams (n_min <= n <= n_max) in a
    Document *x*, divided into Sentences inside of html elements (such as table
    cells).

    :param n_min: Lower limit for the generated n_grams.
    :type n_min: int
    :param n_max: Upper limit for the generated n_grams.
    :type n_max: int
    :param split_tokens: Tokens, on which unigrams are split into two separate
        unigrams.
    :type split_tokens: tuple, list of str.
    """

    def __init__(self, n_min=1, n_max=5, split_tokens=[]):
        """
        Initialize MentionNgrams.
        """
        Ngrams.__init__(self, n_min=n_min, n_max=n_max, split_tokens=split_tokens)

    def apply(self, session, doc):
        """Generate MentionNgrams from a Document by parsing all of its Sentences.

        :param session: The database session
        :param doc: The ``Document`` to parse.
        :type doc: ``Document``.
        :raises TypeError: If the input doc is not of type ``Document``.
        """
        if not isinstance(doc, Document):
            raise TypeError(
                "Input Contexts to MentionNgrams.apply() must be of type Document"
            )

        doc = session.query(Document).filter(Document.id == doc.id).one()
        for sentence in doc.sentences:
            for ts in Ngrams.apply(self, sentence):
                yield ts


class MentionFigures(MentionSpace):
    """Defines the space of Mentions as all figures in a Document *x*, indexing
    by **position offset**.

    :param types: If specified, only yield TemporaryImages whose url ends in
        one of the specified types. Example: types=["png", "jpg", "jpeg"].
    :type types: list, tuple of str
    """

    def __init__(self, types=None):
        """Initialize MentionFigures."""
        MentionSpace.__init__(self)
        if types is not None:
            self.types = [t.strip().lower() for t in types]
        else:
            self.types = None

    def apply(self, session, doc):
        """
        Generate MentionFigures from a Document by parsing all of its Figures.

        :param session: The database session
        :param doc: The ``Document`` to parse.
        :type doc: ``Document``.
        :raises TypeError: If the input doc is not of type ``Document``.
        """
        if not isinstance(doc, Document):
            raise TypeError(
                "Input Contexts to MentionFigures.apply() must be of type Document"
            )

        doc = session.query(Document).filter(Document.id == doc.id).one()
        for figure in doc.figures:
            if self.types is None or any(
                figure.url.lower().endswith(type) for type in self.types
            ):
                yield TemporaryImage(figure)


class MentionExtractor(UDFRunner):
    """An operator to extract Mention objects from a Context.

    :Example:

        Assuming we want to extract two types of ``Mentions``, a Part and a
        Temperature, and we have already defined Matchers to use::

            part_ngrams = MentionNgrams(n_max=3)
            temp_ngrams = MentionNgrams(n_max=2)

            Part = mention_subclass("Part")
            Temp = mention_subclass("Temp")

            mention_extractor = MentionExtractor(
                session,
                [Part, Temp],
                [part_ngrams, temp_ngrams],
                [part_matcher, temp_matcher]
            )

    :param session: An initialized database session.
    :param mention_classes: The type of relation to extract, defined using
        :func: fonduer.mentions.mention_subclass.
    :type mention_classes: list
    :param mention_spaces: one or list of :class:`MentionSpace` objects, one for
        each relation argument. Defines space of Contexts to consider
    :type mention_spaces: list
    :param matchers: one or list of :class:`fonduer.matchers.Matcher` objects,
        one for each relation argument. Only tuples of Contexts for which each
        element is accepted by the corresponding Matcher will be returned as
        Mentions
    :type matchers: list
    :param parallelism: The number of processes to use in parallel for calls
        to apply().
    :type parallelism: int
    :raises ValueError: If mention classes, spaces, and matchers are not the
        same length.
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

    def apply(self, docs, clear=True, parallelism=None, progress_bar=True):
        """Run the MentionExtractor.

        :Example: To extract mentions from a set of training documents using
            4 cores::

                mention_extractor.apply(train_docs, parallelism=4)

        :param docs: Set of documents to extract from.
        :param clear: Whether or not to clear the existing Mentions
            beforehand.
        :type clear: bool
        :param parallelism: How many threads to use for extraction. This will
            override the parallelism value used to initialize the
            MentionExtractor if it is provided.
        :type parallelism: int
        :param progress_bar: Whether or not to display a progress bar. The
            progress bar is measured per document.
        :type progress_bar: bool
        """
        super(MentionExtractor, self).apply(
            docs, clear=clear, parallelism=parallelism, progress_bar=progress_bar
        )

    def clear(self):
        """Delete Mentions of each class in the extractor from the given split."""
        for mention_class in self.mention_classes:
            logger.info("Clearing table: {}".format(mention_class.__tablename__))
            self.session.query(Mention).filter(
                Mention.type == mention_class.__tablename__
            ).delete()

    def clear_all(self):
        """Delete all Mentions from given split the database."""
        logger.info("Clearing ALL Mentions.")
        self.session.query(Mention).delete()

    def get_mentions(self, docs=None):
        """Return a list of lists of the mentions associated with this extractor.

        Each list of the return will contain the Mentions for one of the
        mention classes associated with the MentionExtractor.

        :param docs: If provided, return Mentions from these documents. Else,
            return all Mentions.
        :return: Mentions for each mention_class.
        :rtype: List of lists.
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
            tc_to_insert = defaultdict(list)
            # Generate TemporaryContexts that are children of the context using
            # the mention_space and filtered by the Matcher
            self.child_context_set.clear()
            for tc in self.matchers[i].apply(
                self.mention_spaces[i].apply(self.session, context)
            ):
                rec = tc._load_id_or_insert(self.session)
                if rec:
                    tc_to_insert[tc._get_table()].append(rec)
                self.child_context_set.add(tc)

            # Bulk insert temporary contexts
            for table, records in tc_to_insert.items():
                stmt = insert(table.__table__).values(records)
                self.session.execute(stmt)

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
