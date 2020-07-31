"""Fonduer implicit span mention model."""
from typing import Any, Dict, List, Optional, Type

from sqlalchemy import Column, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import backref, relationship
from sqlalchemy.sql import text
from sqlalchemy.types import PickleType

from fonduer.candidates.models.span_mention import TemporarySpanMention
from fonduer.parser.models.context import Context
from fonduer.parser.models.sentence import Sentence
from fonduer.parser.models.utils import split_stable_id


class TemporaryImplicitSpanMention(TemporarySpanMention):
    """The TemporaryContext version of ImplicitSpanMention."""

    def __init__(
        self,
        sentence: Sentence,
        char_start: int,
        char_end: int,
        expander_key: str,
        position: int,
        text: str,
        words: List[str],
        lemmas: List[str],
        pos_tags: List[str],
        ner_tags: List[str],
        dep_parents: List[int],
        dep_labels: List[str],
        page: List[Optional[int]],
        top: List[Optional[int]],
        left: List[Optional[int]],
        bottom: List[Optional[int]],
        right: List[Optional[int]],
        meta: Any = None,
    ) -> None:
        """Initialize TemporaryImplicitSpanMention."""
        super().__init__(sentence, char_start, char_end, meta)
        self.expander_key = expander_key
        self.position = position
        self.text = text
        self.words = words
        self.lemmas = lemmas
        self.pos_tags = pos_tags
        self.ner_tags = ner_tags
        self.dep_parents = dep_parents
        self.dep_labels = dep_labels
        self.page = page
        self.top = top
        self.left = left
        self.bottom = bottom
        self.right = right

    def __len__(self) -> int:
        """Get the length of the mention."""
        return sum(map(len, self.words))

    def __eq__(self, other: object) -> bool:
        """Check if the mention is equal to another mention."""
        if not isinstance(other, TemporaryImplicitSpanMention):
            return NotImplemented
        return (
            self.sentence == other.sentence
            and self.char_start == other.char_start
            and self.char_end == other.char_end
            and self.expander_key == other.expander_key
            and self.position == other.position
        )

    def __ne__(self, other: object) -> bool:
        """Check if the mention is not equal to another mention."""
        if not isinstance(other, TemporaryImplicitSpanMention):
            return NotImplemented
        return (
            self.sentence != other.sentence
            or self.char_start != other.char_start
            or self.char_end != other.char_end
            or self.expander_key != other.expander_key
            or self.position != other.position
        )

    def __hash__(self) -> int:
        """Get the hash value of mention."""
        return (
            hash(self.sentence)
            + hash(self.char_start)
            + hash(self.char_end)
            + hash(self.expander_key)
            + hash(self.position)
        )

    def get_stable_id(self) -> str:
        """Return a stable id."""
        doc_id, _, idx = split_stable_id(self.sentence.stable_id)
        parent_doc_char_start = idx[0]
        return (
            f"{self.sentence.document.name}"
            f"::"
            f"{self._get_polymorphic_identity()}"
            f":"
            f"{parent_doc_char_start + self.char_start}"
            f":"
            f"{parent_doc_char_start + self.char_end}"
            f":"
            f"{self.expander_key}"
            f":"
            f"{self.position}"
        )

    def _get_table(self) -> Type["ImplicitSpanMention"]:
        return ImplicitSpanMention

    def _get_polymorphic_identity(self) -> str:
        return "implicit_span_mention"

    def _get_insert_args(self) -> Dict[str, Any]:
        return {
            "sentence_id": self.sentence.id,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "expander_key": self.expander_key,
            "position": self.position,
            "text": self.text,
            "words": self.words,
            "lemmas": self.lemmas,
            "pos_tags": self.pos_tags,
            "ner_tags": self.ner_tags,
            "dep_parents": self.dep_parents,
            "dep_labels": self.dep_labels,
            "page": self.page,
            "top": self.top,
            "left": self.left,
            "bottom": self.bottom,
            "right": self.right,
            "meta": self.meta,
        }

    def get_attrib_tokens(self, a: str = "words") -> List:
        """Get the tokens of sentence attribute *a*.

        Intuitively, like calling::

            implicit_span.a


        :param a: The attribute to get tokens for.
        :return: The tokens of sentence attribute defined by *a* for the span.
        """
        return self.__getattribute__(a)

    def get_attrib_span(self, a: str, sep: str = "") -> str:
        """Get the span of sentence attribute *a*.

        Intuitively, like calling::

            sep.join(implicit_span.a)

        :param a: The attribute to get a span for.
        :param sep: The separator to use for the join,
                    or to be removed from text if a="words".
        :return: The joined tokens, or text if a="words".
        """
        if a == "words":
            return self.text.replace(sep, "")
        else:
            return sep.join([str(n) for n in self.get_attrib_tokens(a)])

    def __getitem__(self, key: slice) -> "TemporaryImplicitSpanMention":
        """Slice operation returns a new candidate sliced according to **char index**.

        Note that the slicing is w.r.t. the candidate range (not the abs.
        sentence char indexing)
        """
        if isinstance(key, slice):
            char_start = (
                self.char_start if key.start is None else self.char_start + key.start
            )
            if key.stop is None:
                char_end = self.char_end
            elif key.stop >= 0:
                char_end = self.char_start + key.stop - 1
            else:
                char_end = self.char_end + key.stop
            return self._get_instance(
                sentence=self.sentence,
                char_start=char_start,
                char_end=char_end,
                expander_key=self.expander_key,
                position=self.position,
                text=text,
                words=self.words,
                lemmas=self.lemmas,
                pos_tags=self.pos_tags,
                ner_tags=self.ner_tags,
                dep_parents=self.dep_parents,
                dep_labels=self.dep_labels,
                page=self.page,
                top=self.top,
                left=self.left,
                bottom=self.bottom,
                right=self.right,
                meta=self.meta,
            )
        else:
            raise NotImplementedError()

    def __repr__(self) -> str:
        """Represent the mention as a string."""
        return (
            f"{self.__class__.__name__}"
            f"("
            f'"{self.get_span()}", '
            f"sentence={self.sentence.id}, "
            f"words=[{self.get_word_start_index()},{self.get_word_end_index()}], "
            f"position=[{self.position}]"
            f")"
        )

    def _get_instance(self, **kwargs: Any) -> "TemporaryImplicitSpanMention":
        return TemporaryImplicitSpanMention(**kwargs)


class ImplicitSpanMention(Context, TemporaryImplicitSpanMention):
    """A span of characters that may not appear verbatim in the source text.

    It is identified by Context id, character-index start and end (inclusive),
    as well as a key representing what 'expander' function drew the ImplicitSpanMention
    from an existing SpanMention, and a position (where position=0 corresponds to the
    first ImplicitSpanMention produced from the expander function).

    The character-index start and end point to the segment of text that was
    expanded to produce the ImplicitSpanMention.
    """

    __tablename__ = "implicit_span_mention"

    #: The unique id of the ``ImplicitSpanMention``.
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)

    #: The id of the parent ``Sentence``.
    sentence_id = Column(
        Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True
    )
    #: The parent ``Sentence``.
    sentence = relationship(
        "Context",
        backref=backref("implicit_spans", cascade="all, delete-orphan"),
        foreign_keys=sentence_id,
    )

    #: The starting character-index of the ``ImplicitSpanMention``.
    char_start = Column(Integer, nullable=False)

    #: The ending character-index of the ``ImplicitSpanMention`` (inclusive).
    char_end = Column(Integer, nullable=False)

    #: The key representing the expander function which produced this
    # ``ImplicitSpanMention``.
    expander_key = Column(String, nullable=False)

    #: The position of the ``ImplicitSpanMention`` where position=0 is the first
    #: ``ImplicitSpanMention`` produced by the expander.
    position = Column(Integer, nullable=False)

    #: The raw text of the ``ImplicitSpanMention``.
    text = Column(String)

    #: A list of the words in the ``ImplicitSpanMention``.
    words = Column(postgresql.ARRAY(String), nullable=False)

    #: A list of the lemmas for each word in the ``ImplicitSpanMention``.
    lemmas = Column(postgresql.ARRAY(String))

    #: A list of the POS tags for each word in the ``ImplicitSpanMention``.
    pos_tags = Column(postgresql.ARRAY(String))

    #: A list of the NER tags for each word in the ``ImplicitSpanMention``.
    ner_tags = Column(postgresql.ARRAY(String))

    #: A list of the dependency parents for each word in the ``ImplicitSpanMention``.
    dep_parents = Column(postgresql.ARRAY(Integer))

    #: A list of the dependency labels for each word in the ``ImplicitSpanMention``.
    dep_labels = Column(postgresql.ARRAY(String))

    #: A list of the page number each word in the ``ImplicitSpanMention``.
    page = Column(postgresql.ARRAY(Integer))

    #: A list of each word's TOP bounding box coordinate in the
    # ``ImplicitSpanMention``.
    top = Column(postgresql.ARRAY(Integer))

    #: A list of each word's LEFT bounding box coordinate in the
    # ``ImplicitSpanMention``.
    left = Column(postgresql.ARRAY(Integer))

    #: A list of each word's BOTTOM bounding box coordinate in the
    # ``ImplicitSpanMention``.
    bottom = Column(postgresql.ARRAY(Integer))

    #: A list of each word's RIGHT bounding box coordinate in the
    # ``ImplicitSpanMention``.
    right = Column(postgresql.ARRAY(Integer))

    #: Pickled metadata about the ``ImplicitSpanMention``.
    meta = Column(PickleType)

    __table_args__ = (
        UniqueConstraint(sentence_id, char_start, char_end, expander_key, position),
    )

    __mapper_args__ = {
        "polymorphic_identity": "implicit_span_mention",
        "inherit_condition": (id == Context.id),
    }

    def __init__(self, tc: TemporaryImplicitSpanMention):
        """Initialize ImplicitSpanMention."""
        self.stable_id = tc.get_stable_id()
        self.sentence = tc.sentence
        self.char_start = tc.char_start
        self.char_end = tc.char_end
        self.expander_key = tc.expander_key
        self.position = tc.position
        self.text = tc.text
        self.words = tc.words
        self.lemmas = tc.lemmas
        self.pos_tags = tc.pos_tags
        self.ner_tags = tc.ner_tags
        self.dep_parents = tc.dep_parents
        self.dep_labels = tc.dep_labels
        self.page = tc.page
        self.top = tc.top
        self.left = tc.left
        self.bottom = tc.bottom
        self.right = tc.right
        self.meta = tc.meta

    def _get_instance(self, **kwargs: Any) -> "ImplicitSpanMention":
        return ImplicitSpanMention(**kwargs)

    # We redefine these to use default semantics, overriding the operators
    # inherited from TemporarySpan
    def __eq__(self, other: object) -> bool:
        """Check if the mention is equal to another mention."""
        if not isinstance(other, ImplicitSpanMention):
            return NotImplemented
        return self is other

    def __ne__(self, other: object) -> bool:
        """Check if the mention is not equal to another mention."""
        if not isinstance(other, ImplicitSpanMention):
            return NotImplemented
        return self is not other

    def __hash__(self) -> int:
        """Get the hash value of mention."""
        return id(self)
