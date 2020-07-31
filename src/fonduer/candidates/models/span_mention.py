"""Fonduer span mention model."""
from typing import Any, Dict, List, Optional, Type

from sqlalchemy import Column, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import backref, relationship
from sqlalchemy.types import PickleType

from fonduer.candidates.models.temporary_context import TemporaryContext
from fonduer.parser.models.context import Context
from fonduer.parser.models.sentence import Sentence
from fonduer.parser.models.utils import construct_stable_id
from fonduer.utils.utils_visual import Bbox


class TemporarySpanMention(TemporaryContext):
    """The TemporaryContext version of Span."""

    def __init__(
        self,
        sentence: Sentence,
        char_start: int,
        char_end: int,
        meta: Optional[Any] = None,
    ) -> None:
        """Initialize TemporarySpanMention."""
        super().__init__()
        self.sentence = sentence  # The sentence Context of the Span
        self.char_start = char_start
        self.char_end = char_end
        self.meta = meta

    def __len__(self) -> int:
        """Get the length of the mention."""
        return self.char_end - self.char_start + 1

    def __eq__(self, other: object) -> bool:
        """Check if the mention is equal to another mention."""
        if not isinstance(other, TemporarySpanMention):
            return NotImplemented
        return (
            self.sentence == other.sentence
            and self.char_start == other.char_start
            and self.char_end == other.char_end
        )

    def __ne__(self, other: object) -> bool:
        """Check if the mention is not equal to another mention."""
        if not isinstance(other, TemporarySpanMention):
            return NotImplemented
        return (
            self.sentence != other.sentence
            or self.char_start != other.char_start
            or self.char_end != other.char_end
        )

    def __hash__(self) -> int:
        """Get the hash value of mention."""
        return hash(self.sentence) + hash(self.char_start) + hash(self.char_end)

    def get_stable_id(self) -> str:
        """Return a stable id."""
        return construct_stable_id(
            self.sentence,
            self._get_polymorphic_identity(),
            self.char_start,
            self.char_end,
        )

    def _get_table(self) -> Type["SpanMention"]:
        return SpanMention

    def _get_polymorphic_identity(self) -> str:
        return "span_mention"

    def _get_insert_args(self) -> Dict[str, Any]:
        return {
            "sentence_id": self.sentence.id,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "meta": self.meta,
        }

    def get_word_start_index(self) -> int:
        """Get the index of the starting word of the span.

        :return: The word-index of the start of the span.
        """
        return self._char_to_word_index(self.char_start)

    def get_word_end_index(self) -> int:
        """Get the index of the ending word of the span.

        :return: The word-index of the last word of the span.
        """
        return self._char_to_word_index(self.char_end)

    def get_num_words(self) -> int:
        """Get the number of words in the span.

        :return: The number of words in the span (n of the ngrams).
        """
        return self.get_word_end_index() - self.get_word_start_index() + 1

    def _char_to_word_index(self, ci: int) -> int:
        """Return the index of the **word this char is in**.

        :param ci: The character-level index of the char.
        :return: The word-level index the char was in.
        """
        i = None
        for i, co in enumerate(self.sentence.char_offsets):
            if ci == co:
                return i
            elif ci < co:
                return i - 1
        return i

    def _word_to_char_index(self, wi: int) -> int:
        """Return the character-level index (offset) of the word's start.

        :param wi: The word-index.
        :return: The character-level index of the word's start.
        """
        return self.sentence.char_offsets[wi]

    def get_attrib_tokens(self, a: str = "words") -> List:
        """Get the tokens of sentence attribute *a*.

        Intuitively, like calling::

            span.a


        :param a: The attribute to get tokens for.
        :return: The tokens of sentence attribute defined by *a* for the span.
        """
        return self.sentence.__getattribute__(a)[
            self.get_word_start_index() : self.get_word_end_index() + 1
        ]

    def get_attrib_span(self, a: str, sep: str = "") -> str:
        """Get the span of sentence attribute *a*.

        Intuitively, like calling::

            sep.join(span.a)

        :param a: The attribute to get a span for.
        :param sep: The separator to use for the join,
                    or to be removed from text if a="words".
        :return: The joined tokens, or text if a="words".
        """
        # NOTE: Special behavior for words currently (due to correspondence
        # with char_offsets)
        if a == "words":
            return self.sentence.text[self.char_start : self.char_end + 1].replace(
                sep, ""
            )
        else:
            return sep.join([str(n) for n in self.get_attrib_tokens(a)])

    def get_span(self) -> str:
        """Return the text of the ``Span``.

        :return: The text of the ``Span``.
        """
        return self.get_attrib_span("words")

    def get_bbox(self) -> Bbox:
        """Get the bounding box."""
        if self.sentence.is_visual():
            return Bbox(
                self.get_attrib_tokens("page")[0],
                min(self.get_attrib_tokens("top")),
                max(self.get_attrib_tokens("bottom")),
                min(self.get_attrib_tokens("left")),
                max(self.get_attrib_tokens("right")),
            )
        else:
            return None

    def __contains__(self, other_span: object) -> bool:
        """Check if the mention contains another mention."""
        if not isinstance(other_span, TemporarySpanMention):
            return NotImplemented
        return (
            self.sentence == other_span.sentence
            and other_span.char_start >= self.char_start
            and other_span.char_end <= self.char_end
        )

    def __getitem__(self, key: slice) -> "TemporarySpanMention":
        """Slice operation returns a new candidate sliced according to **char index**.

        Note that the slicing is w.r.t. the candidate range (not the abs.
        sentence char indexing).
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
                char_start=char_start, char_end=char_end, sentence=self.sentence
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
            f"chars=[{self.char_start},{self.char_end}], "
            f"words=[{self.get_word_start_index()},{self.get_word_end_index()}]"
            f")"
        )

    def _get_instance(self, **kwargs: Any) -> "TemporarySpanMention":
        return TemporarySpanMention(**kwargs)


class SpanMention(Context, TemporarySpanMention):
    """
    A span of chars, identified by Context ID and char-index start, end (inclusive).

    char_offsets are **relative to the Context start**
    """

    __tablename__ = "span_mention"

    #: The unique id of the ``SpanMention``.
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)

    #: The id of the parent ``Sentence``.
    sentence_id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"))

    #: The parent ``Sentence``.
    sentence = relationship(
        "Context",
        backref=backref("spans", cascade="all, delete-orphan"),
        foreign_keys=sentence_id,
    )

    #: The starting character-index of the ``SpanMention``.
    char_start = Column(Integer, nullable=False)

    #: The ending character-index of the ``SpanMention`` (inclusive).
    char_end = Column(Integer, nullable=False)

    #: Pickled metadata about the ``ImplicitSpanMention``.
    meta = Column(PickleType)

    __table_args__ = (UniqueConstraint(sentence_id, char_start, char_end),)

    __mapper_args__ = {
        "polymorphic_identity": "span_mention",
        "inherit_condition": (id == Context.id),
    }

    def __init__(self, tc: TemporarySpanMention):
        """Initialize SpanMention."""
        self.stable_id = tc.get_stable_id()
        self.sentence = tc.sentence
        self.char_start = tc.char_start
        self.char_end = tc.char_end
        self.meta = tc.meta

    def _get_instance(self, **kwargs: Any) -> "SpanMention":
        return SpanMention(**kwargs)

    # We redefine these to use default semantics, overriding the operators
    # inherited from TemporarySpanMention
    def __eq__(self, other: object) -> bool:
        """Check if the mention is equal to another mention."""
        if not isinstance(other, SpanMention):
            return NotImplemented
        return self is other

    def __ne__(self, other: object) -> bool:
        """Check if the mention is not equal to another mention."""
        if not isinstance(other, SpanMention):
            return NotImplemented
        return self is not other

    def __hash__(self) -> int:
        """Get the hash value of mention."""
        return id(self)
