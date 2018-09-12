from sqlalchemy import Column, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import backref, relationship
from sqlalchemy.types import PickleType

from fonduer.candidates.models.temporarycontext import TemporaryContext
from fonduer.parser.models.context import Context
from fonduer.parser.models.utils import construct_stable_id


class TemporarySpan(TemporaryContext):
    """The TemporaryContext version of Span"""

    def __init__(self, sentence, char_start, char_end, meta=None):
        super(TemporarySpan, self).__init__()
        self.sentence = sentence  # The sentence Context of the Span
        self.char_end = char_end
        self.char_start = char_start
        self.meta = meta

    def __len__(self):
        return self.char_end - self.char_start + 1

    def __eq__(self, other):
        try:
            return (
                self.sentence == other.sentence
                and self.char_start == other.char_start
                and self.char_end == other.char_end
            )
        except AttributeError:
            return False

    def __ne__(self, other):
        try:
            return (
                self.sentence != other.sentence
                or self.char_start != other.char_start
                or self.char_end != other.char_end
            )
        except AttributeError:
            return True

    def __hash__(self):
        return hash(self.sentence) + hash(self.char_start) + hash(self.char_end)

    def get_stable_id(self):
        return construct_stable_id(
            self.sentence,
            self._get_polymorphic_identity(),
            self.char_start,
            self.char_end,
        )

    def _get_table_name(self):
        return "span"

    def _get_polymorphic_identity(self):
        return "span"

    def _get_insert_query(self):
        return (
            "INSERT INTO span VALUES"
            + "(:id, :sentence_id, :char_start, :char_end, :meta)"
        )

    def _get_insert_args(self):
        return {
            "sentence_id": self.sentence.id,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "meta": self.meta,
        }

    def get_word_start(self):
        return self.char_to_word_index(self.char_start)

    def get_word_end(self):
        return self.char_to_word_index(self.char_end)

    def get_n(self):
        return self.get_word_end() - self.get_word_start() + 1

    def char_to_word_index(self, ci):
        """Return the index of the **word this char is in**"""
        i = None
        for i, co in enumerate(self.sentence.char_offsets):
            if ci == co:
                return i
            elif ci < co:
                return i - 1
        return i

    def word_to_char_index(self, wi):
        """Return the character-level index (offset) of the word's start"""
        return self.sentence.char_offsets[wi]

    def get_attrib_tokens(self, a="words"):
        """Get the tokens of sentence attribute *a*."""
        return self.sentence.__getattribute__(a)[
            self.get_word_start() : self.get_word_end() + 1
        ]

    def get_attrib_span(self, a, sep=" "):
        """Get the span of sentence attribute *a*."""
        # NOTE: Special behavior for words currently (due to correspondence
        # with char_offsets)
        if a == "words":
            return self.sentence.text[self.char_start : self.char_end + 1]
        else:
            return sep.join(self.get_attrib_tokens(a))

    def get_span(self, sep=" "):
        return self.get_attrib_span("words", sep)

    def __contains__(self, other_span):
        return (
            self.sentence == other_span.sentence
            and other_span.char_start >= self.char_start
            and other_span.char_end <= self.char_end
        )

    def __getitem__(self, key):
        """
        Slice operation returns a new candidate sliced according to **char index**

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

    def __repr__(self):
        return '{}("{}", sentence={}, chars=[{},{}], words=[{},{}])'.format(
            self.__class__.__name__,
            self.get_span(),
            self.sentence.id,
            self.char_start,
            self.char_end,
            self.get_word_start(),
            self.get_word_end(),
        )

    def _get_instance(self, **kwargs):
        return TemporarySpan(**kwargs)


class Span(Context, TemporarySpan):
    """
    A span of chars, identified by Context ID and char-index start, end (inclusive).

    char_offsets are **relative to the Context start**
    """

    __tablename__ = "span"
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)
    sentence_id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"))
    char_start = Column(Integer, nullable=False)
    char_end = Column(Integer, nullable=False)
    meta = Column(PickleType)

    __table_args__ = (UniqueConstraint(sentence_id, char_start, char_end),)

    __mapper_args__ = {
        "polymorphic_identity": "span",
        "inherit_condition": (id == Context.id),
    }

    sentence = relationship(
        "Context",
        backref=backref("spans", cascade="all, delete-orphan"),
        foreign_keys=sentence_id,
    )

    def _get_instance(self, **kwargs):
        return Span(**kwargs)

    # We redefine these to use default semantics, overriding the operators
    # inherited from TemporarySpan
    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other

    def __hash__(self):
        return id(self)
