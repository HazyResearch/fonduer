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
        """
        Return a stable id.

        :rtype: string
        """
        return construct_stable_id(
            self.sentence,
            self._get_polymorphic_identity(),
            self.char_start,
            self.char_end,
        )

    def _get_table(self):
        return Span

    def _get_polymorphic_identity(self):
        return "span"

    def _get_insert_args(self):
        return {
            "sentence_id": self.sentence.id,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "meta": self.meta,
        }

    def get_word_start_index(self):
        """Get the index of the starting word of the span.

        :return: The word-index of the start of the span.
        :rtype: int
        """
        return self._char_to_word_index(self.char_start)

    def get_word_end_index(self):
        """Get the index of the ending word of the span.

        :return: The word-index of the last word of the span.
        :rtype: int
        """
        return self._char_to_word_index(self.char_end)

    def get_num_words(self):
        """Get the number of words in the span.

        :return: The number of words in the span (n of the ngrams).
        :rtype: int
        """
        return self.get_word_end_index() - self.get_word_start_index() + 1

    def _char_to_word_index(self, ci):
        """Return the index of the **word this char is in**.

        :param ci: The character-level index of the char.
        :type ci: int
        :return: The word-level index the char was in.
        :rtype: int
        """
        i = None
        for i, co in enumerate(self.sentence.char_offsets):
            if ci == co:
                return i
            elif ci < co:
                return i - 1
        return i

    def _word_to_char_index(self, wi):
        """Return the character-level index (offset) of the word's start.

        :param wi: The word-index.
        :type wi: int
        :return: The character-level index of the word's start.
        :rtype: int
        """
        return self.sentence.char_offsets[wi]

    def get_attrib_tokens(self, a="words"):
        """Get the tokens of sentence attribute *a*.

        Intuitively, like calling::

            span.a


        :param a: The attribute to get tokens for.
        :type a: str
        :return: The tokens of sentence attribute defined by *a* for the span.
        :rtype: list
        """
        return self.sentence.__getattribute__(a)[
            self.get_word_start_index() : self.get_word_end_index() + 1
        ]

    def get_attrib_span(self, a, sep=" "):
        """Get the span of sentence attribute *a*.

        Intuitively, like calling::

            sep.join(span.a)

        :param a: The attribute to get a span for.
        :type a: str
        :param sep: The separator to use for the join.
        :type sep: str
        :return: The joined tokens, or text if a="words".
        :rtype: str
        """
        # NOTE: Special behavior for words currently (due to correspondence
        # with char_offsets)
        if a == "words":
            return self.sentence.text[self.char_start : self.char_end + 1]
        else:
            return sep.join(self.get_attrib_tokens(a))

    def get_span(self):
        """Return the text of the ``Span``.

        :return: The text of the ``Span``.
        :rtype: str
        """
        return self.get_attrib_span("words")

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
            self.get_word_start_index(),
            self.get_word_end_index(),
        )

    def _get_instance(self, **kwargs):
        return TemporarySpan(**kwargs)


class Span(Context, TemporarySpan):
    """
    A span of chars, identified by Context ID and char-index start, end (inclusive).

    char_offsets are **relative to the Context start**
    """

    __tablename__ = "span"

    #: The unique id of the ``Span``.
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)

    #: The id of the parent ``Sentence``.
    sentence_id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"))
    #: The parent ``Sentence``.
    sentence = relationship(
        "Context",
        backref=backref("spans", cascade="all, delete-orphan"),
        foreign_keys=sentence_id,
    )

    #: The starting character-index of the ``Span``.
    char_start = Column(Integer, nullable=False)
    #: The ending character-index of the ``Span`` (inclusive).
    char_end = Column(Integer, nullable=False)

    #: Pickled metadata about the ``ImplicitSpan``.
    meta = Column(PickleType)

    __table_args__ = (UniqueConstraint(sentence_id, char_start, char_end),)

    __mapper_args__ = {
        "polymorphic_identity": "span",
        "inherit_condition": (id == Context.id),
    }

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
