"""Fonduer mention."""
import logging
import re
from builtins import map, range
from typing import Any, Collection, Dict, Iterable, Iterator, List, Optional, Set, Union

from sqlalchemy.orm import Session

from fonduer.candidates.matchers import _Matcher
from fonduer.candidates.models import Candidate, Mention
from fonduer.candidates.models.candidate import candidate_subclasses
from fonduer.candidates.models.caption_mention import TemporaryCaptionMention
from fonduer.candidates.models.cell_mention import TemporaryCellMention
from fonduer.candidates.models.document_mention import TemporaryDocumentMention
from fonduer.candidates.models.figure_mention import TemporaryFigureMention
from fonduer.candidates.models.paragraph_mention import TemporaryParagraphMention
from fonduer.candidates.models.section_mention import TemporarySectionMention
from fonduer.candidates.models.span_mention import TemporarySpanMention
from fonduer.candidates.models.table_mention import TemporaryTableMention
from fonduer.candidates.models.temporary_context import TemporaryContext
from fonduer.parser.models import Context, Document, Sentence
from fonduer.utils.udf import UDF, UDFRunner
from fonduer.utils.utils import get_dict_of_stable_id

logger = logging.getLogger(__name__)


class MentionSpace(object):
    """Define the **space** of Mention objects.

    Calling *apply(x)* given an object *x* returns a generator over mentions in
    *x*.
    """

    def __init__(self) -> None:
        """Initialize mention space."""
        pass

    def apply(self, x: Context) -> Iterator[TemporaryContext]:
        """Apply function takes a Context and return a mention generator.

        :param x: The input Context.
        :yield: The mention generator.
        """
        raise NotImplementedError()


class Ngrams(MentionSpace):
    """Define the space of Mentions as all n-grams in a Sentence.

    Define the space of Mentions as all n-grams (n_min <= n <= n_max) in a
    Sentence *x*, indexing by **character offset**.

    :param n_min: Lower limit for the generated n_grams.
    :param n_max: Upper limit for the generated n_grams.
    :param split_tokens: Tokens, on which unigrams are split into two separate
        unigrams.
    :type split_tokens: tuple, list of str.
    """

    def __init__(
        self, n_min: int = 1, n_max: int = 5, split_tokens: Collection[str] = []
    ) -> None:
        """Initialize Ngrams."""
        MentionSpace.__init__(self)
        self.n_min = n_min
        self.n_max = n_max
        self.split_rgx = (
            r"(" + r"|".join(map(re.escape, sorted(split_tokens, reverse=True))) + r")"
            if split_tokens and len(split_tokens) > 0
            else None
        )

    def apply(self, context: Sentence) -> Iterator[TemporarySpanMention]:
        """Apply function takes a Sentence and return a mention generator.

        :param x: The input Sentence.
        :yield: The mention generator.
        """
        # These are the character offset--**relative to the sentence
        # start**--for each _token_
        offsets = context.char_offsets

        # Loop over all n-grams in **reverse** order (to facilitate
        # longest-match semantics)
        L = len(offsets)
        seen: Set[TemporarySpanMention] = set()
        for j in range(self.n_min, self.n_max + 1)[::-1]:
            for i in range(L - j + 1):
                w = context.words[i + j - 1]
                start = offsets[i]
                end = offsets[i + j - 1] + len(w) - 1
                ts = TemporarySpanMention(
                    char_start=start, char_end=end, sentence=context
                )
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
                                ts = TemporarySpanMention(
                                    char_start=start_idx,
                                    char_end=end_idx - 1,
                                    sentence=context,
                                )
                                if ts not in seen and ts.get_span():
                                    seen.add(ts)
                                    yield ts


class MentionNgrams(Ngrams):
    """Defines the **space** of Mentions as n-grams in a Document.

    Defines the space of Mentions as all n-grams (n_min <= n <= n_max) in a
    Document *x*, divided into Sentences inside of html elements (such as table
    cells).

    :param n_min: Lower limit for the generated n_grams.
    :param n_max: Upper limit for the generated n_grams.
    :param split_tokens: Tokens, on which unigrams are split into two separate
        unigrams.
    :type split_tokens: tuple, list of str.
    """

    def __init__(
        self, n_min: int = 1, n_max: int = 5, split_tokens: Collection[str] = []
    ) -> None:
        """Initialize MentionNgrams."""
        Ngrams.__init__(self, n_min=n_min, n_max=n_max, split_tokens=split_tokens)

    def apply(self, doc: Document) -> Iterator[TemporarySpanMention]:
        """Generate MentionNgrams from a Document by parsing all of its Sentences.

        :param doc: The ``Document`` to parse.
        :raises TypeError: If the input doc is not of type ``Document``.
        """
        if not isinstance(doc, Document):
            raise TypeError(
                "Input Contexts to MentionNgrams.apply() must be of type Document"
            )

        for sentence in doc.sentences:
            for ts in Ngrams.apply(self, sentence):
                yield ts


class MentionFigures(MentionSpace):
    """Defines the space of Mentions as all figures in a Document *x*.

    :param types: If specified, only yield TemporaryFigureMentions whose url ends in
        one of the specified types. Example: types=["png", "jpg", "jpeg"].
    :type types: list, tuple of str
    """

    def __init__(self, types: Optional[str] = None) -> None:
        """Initialize MentionFigures."""
        MentionSpace.__init__(self)
        if types is not None:
            self.types = [t.strip().lower() for t in types]
        else:
            self.types = None

    def apply(self, doc: Document) -> Iterator[TemporaryFigureMention]:
        """
        Generate MentionFigures from a Document by parsing all of its Figures.

        :param doc: The ``Document`` to parse.
        :raises TypeError: If the input doc is not of type ``Document``.
        """
        if not isinstance(doc, Document):
            raise TypeError(
                "Input Contexts to MentionFigures.apply() must be of type Document"
            )

        for figure in doc.figures:
            if self.types is None or any(
                figure.url.lower().endswith(type) for type in self.types
            ):
                yield TemporaryFigureMention(figure)


class MentionSentences(MentionSpace):
    """Defines the space of Mentions as all sentences in a Document *x*."""

    def __init__(self) -> None:
        """Initialize MentionSentences."""
        MentionSpace.__init__(self)

    def apply(self, doc: Document) -> Iterator[TemporarySpanMention]:
        """
        Generate MentionSentences from a Document by parsing all of its Sentences.

        :param doc: The ``Document`` to parse.
        :raises TypeError: If the input doc is not of type ``Document``.
        """
        if not isinstance(doc, Document):
            raise TypeError(
                "Input Contexts to MentionSentences.apply() must be of type Document"
            )

        for sentence in doc.sentences:
            yield TemporarySpanMention(
                char_start=0, char_end=len(sentence.text) - 1, sentence=sentence
            )


class MentionParagraphs(MentionSpace):
    """Defines the space of Mentions as all paragraphs in a Document *x*."""

    def __init__(self) -> None:
        """Initialize MentionParagraphs."""
        MentionSpace.__init__(self)

    def apply(self, doc: Document) -> Iterator[TemporaryParagraphMention]:
        """
        Generate MentionParagraphs from a Document by parsing all of its Paragraphs.

        :param doc: The ``Document`` to parse.
        :raises TypeError: If the input doc is not of type ``Document``.
        """
        if not isinstance(doc, Document):
            raise TypeError(
                "Input Contexts to MentionParagraphs.apply() must be of type Document"
            )

        for paragraph in doc.paragraphs:
            yield TemporaryParagraphMention(paragraph)


class MentionCaptions(MentionSpace):
    """Defines the space of Mentions as all captions in a Document *x*."""

    def __init__(self) -> None:
        """Initialize MentionCaptions."""
        MentionSpace.__init__(self)

    def apply(self, doc: Document) -> Iterator[TemporaryCaptionMention]:
        """
        Generate MentionCaptions from a Document by parsing all of its Captions.

        :param doc: The ``Document`` to parse.
        :raises TypeError: If the input doc is not of type ``Document``.
        """
        if not isinstance(doc, Document):
            raise TypeError(
                "Input Contexts to MentionCaptions.apply() must be of type Document"
            )

        for caption in doc.captions:
            yield TemporaryCaptionMention(caption)


class MentionCells(MentionSpace):
    """Defines the space of Mentions as all cells in a Document *x*."""

    def __init__(self) -> None:
        """Initialize MentionCells."""
        MentionSpace.__init__(self)

    def apply(self, doc: Document) -> Iterator[TemporaryCellMention]:
        """
        Generate MentionCells from a Document by parsing all of its Cells.

        :param doc: The ``Document`` to parse.
        :raises TypeError: If the input doc is not of type ``Document``.
        """
        if not isinstance(doc, Document):
            raise TypeError(
                "Input Contexts to MentionCells.apply() must be of type Document"
            )

        for cell in doc.cells:
            yield TemporaryCellMention(cell)


class MentionTables(MentionSpace):
    """Defines the space of Mentions as all tables in a Document *x*."""

    def __init__(self) -> None:
        """Initialize MentionTables."""
        MentionSpace.__init__(self)

    def apply(self, doc: Document) -> Iterator[TemporaryTableMention]:
        """
        Generate MentionTables from a Document by parsing all of its Tables.

        :param doc: The ``Document`` to parse.
        :raises TypeError: If the input doc is not of type ``Document``.
        """
        if not isinstance(doc, Document):
            raise TypeError(
                "Input Contexts to MentionTables.apply() must be of type Document"
            )

        for table in doc.tables:
            yield TemporaryTableMention(table)


class MentionSections(MentionSpace):
    """Defines the space of Mentions as all sections in a Document *x*."""

    def __init__(self) -> None:
        """Initialize MentionSections."""
        MentionSpace.__init__(self)

    def apply(self, doc: Document) -> Iterator[TemporarySectionMention]:
        """
        Generate MentionSections from a Document by parsing all of its Sections.

        :param doc: The ``Document`` to parse.
        :raises TypeError: If the input doc is not of type ``Document``.
        """
        if not isinstance(doc, Document):
            raise TypeError(
                "Input Contexts to MentionSections.apply() must be of type Document"
            )

        for section in doc.sections:
            yield TemporarySectionMention(section)


class MentionDocuments(MentionSpace):
    """Defines the space of Mentions as a document in a Document *x*."""

    def __init__(self) -> None:
        """Initialize MentionDocuments."""
        MentionSpace.__init__(self)

    def apply(self, doc: Document) -> Iterator[TemporaryDocumentMention]:
        """
        Generate MentionDocuments from a Document by using document.

        :param doc: The ``Document`` to parse.
        :raises TypeError: If the input doc is not of type ``Document``.
        """
        if not isinstance(doc, Document):
            raise TypeError(
                "Input Contexts to MentionDocuments.apply() must be of type Document"
            )

        yield TemporaryDocumentMention(doc)


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
    :param mention_spaces: one or list of :class:`MentionSpace` objects, one for
        each relation argument. Defines space of Contexts to consider
    :param matchers: one or list of :class:`fonduer.matchers.Matcher` objects,
        one for each relation argument. Only tuples of Contexts for which each
        element is accepted by the corresponding Matcher will be returned as
        Mentions
    :param parallelism: The number of processes to use in parallel for calls
        to apply().
    :raises ValueError: If mention classes, spaces, and matchers are not the
        same length.
    """

    def __init__(
        self,
        session: Session,
        mention_classes: List[Mention],
        mention_spaces: List[MentionSpace],
        matchers: List[_Matcher],
        parallelism: int = 1,
    ):
        """Initialize the MentionExtractor."""
        super().__init__(
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
            len(x) == arity  # type: ignore
            for x in [mention_classes, mention_spaces, matchers]
        ):
            raise ValueError(
                "Mismatched arity of mention classes, spaces, and matchers."
            )

        self.mention_classes = mention_classes

    def apply(  # type: ignore
        self,
        docs: Collection[Document],
        clear: bool = True,
        parallelism: Optional[int] = None,
        progress_bar: bool = True,
    ) -> None:
        """Run the MentionExtractor.

        :Example: To extract mentions from a set of training documents using
            4 cores::

                mention_extractor.apply(train_docs, parallelism=4)

        :param docs: Set of documents to extract from.
        :param clear: Whether or not to clear the existing Mentions
            beforehand.
        :param parallelism: How many threads to use for extraction. This will
            override the parallelism value used to initialize the
            MentionExtractor if it is provided.
        :param progress_bar: Whether or not to display a progress bar. The
            progress bar is measured per document.
        """
        super().apply(
            docs, clear=clear, parallelism=parallelism, progress_bar=progress_bar
        )

    def clear(self) -> None:  # type: ignore
        """Delete Mentions of each class in the extractor from the given split."""
        # Create set of candidate_subclasses associated with each mention_subclass
        cand_subclasses = set()
        for mentions, tablename in [
            (_[1][0], _[1][1]) for _ in candidate_subclasses.values()
        ]:
            for mention in mentions:
                if mention in self.mention_classes:
                    cand_subclasses.add(tablename)

        # First, clear all the Mentions. This will cascade and remove the
        # mention_subclasses and corresponding candidate_subclasses.
        for mention_class in self.mention_classes:
            logger.info(f"Clearing table: {mention_class.__tablename__}")
            self.session.query(Mention).filter_by(
                type=mention_class.__tablename__
            ).delete(synchronize_session="fetch")

        # Next, clear the Candidates. This is done manually because we have
        # no cascading relationship from candidate_subclass to Candidate.
        for cand_subclass in cand_subclasses:
            logger.info(f"Cascading to clear table: {cand_subclass}")
            self.session.query(Candidate).filter_by(type=cand_subclass).delete(
                synchronize_session="fetch"
            )

    def clear_all(self) -> None:
        """Delete all Mentions from given split the database."""
        logger.info("Clearing ALL Mentions.")
        self.session.query(Mention).delete(synchronize_session="fetch")

        # With no Mentions, there should be no Candidates also
        self.session.query(Candidate).delete(synchronize_session="fetch")
        logger.info("Cleared ALL Mentions (and Candidates).")

    def get_mentions(
        self, docs: Union[Document, Iterable[Document], None] = None, sort: bool = False
    ) -> List[List[Mention]]:
        """Return a list of lists of the mentions associated with this extractor.

        Each list of the return will contain the Mentions for one of the
        mention classes associated with the MentionExtractor.

        :param docs: If provided, return Mentions from these documents. Else,
            return all Mentions.
        :param sort: If sort is True, then return all Mentions sorted by stable_id.
        :return: Mentions for each mention_class.
        """
        result = []
        if docs:
            docs = docs if isinstance(docs, Iterable) else [docs]
            # Get cands from all splits
            for mention_class in self.mention_classes:
                mentions = (
                    self.session.query(mention_class)
                    .filter(mention_class.document_id.in_([doc.id for doc in docs]))
                    .order_by(mention_class.id)
                    .all()
                )
                if sort:
                    mentions = sorted(mentions, key=lambda x: x[0].get_stable_id())
                result.append(mentions)
        else:
            for mention_class in self.mention_classes:
                mentions = (
                    self.session.query(mention_class).order_by(mention_class.id).all()
                )
                if sort:
                    mentions = sorted(mentions, key=lambda x: x[0].get_stable_id())
                result.append(mentions)
        return result


class MentionExtractorUDF(UDF):
    """UDF for performing mention extraction."""

    def __init__(
        self,
        mention_classes: Union[Mention, List[Mention]],
        mention_spaces: Union[MentionSpace, List[MentionSpace]],
        matchers: Union[_Matcher, List[_Matcher]],
        **kwargs: Any,
    ):
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

        # Preallocates internal data structure
        self.child_context_set: Set[TemporaryContext] = set()

        super().__init__(**kwargs)

    def apply(self, doc: Document, **kwargs: Any) -> Document:
        """Extract mentions from the given Document.

        :param doc: A document to process.
        """
        # Get a dict of stable_id of contexts.
        dict_of_stable_id: Dict[str, Context] = get_dict_of_stable_id(doc)

        # Iterate over each mention class
        for i, mention_class in enumerate(self.mention_classes):
            # Generate TemporaryContexts that are children of the context using
            # the mention_space and filtered by the Matcher
            for child_context in self.matchers[i].apply(
                self.mention_spaces[i].apply(doc)
            ):
                # Skip if this temporary context is used by this mention class.
                stable_id = child_context.get_stable_id()
                if hasattr(doc, mention_class.__tablename__ + "s") and any(
                    [
                        m.context.stable_id == stable_id
                        for m in getattr(doc, mention_class.__tablename__ + "s")
                    ]
                ):
                    continue
                # Re-use a persisted context if exists.
                if stable_id in dict_of_stable_id:
                    context = dict_of_stable_id[stable_id]
                # Persist a temporary context.
                else:
                    context_type = child_context._get_table()
                    context = context_type(child_context)
                    dict_of_stable_id[stable_id] = context

                mention_args = {"document": doc, "context": context}

                # Add Mention to session
                mention_class(**mention_args)
        return doc
