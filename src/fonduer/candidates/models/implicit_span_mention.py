from sqlalchemy import Column, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import relationship
from sqlalchemy.sql import text
from sqlalchemy.types import PickleType

from fonduer.candidates.models.span_mention import TemporarySpanMention
from fonduer.parser.models.context import Context
from fonduer.parser.models.utils import split_stable_id


class TemporaryImplicitSpanMention(TemporarySpanMention):
    """The TemporaryContext version of ImplicitSpanMention."""

    def __init__(
        self,
        sentence,
        char_start,
        char_end,
        expander_key,
        position,
        text,
        words,
        lemmas,
        pos_tags,
        ner_tags,
        dep_parents,
        dep_labels,
        page,
        top,
        left,
        bottom,
        right,
        meta=None,
    ):
        super(TemporarySpanMention, self).__init__()
        self.sentence = sentence  # The sentence Context of the Span
        self.char_start = char_start
        self.char_end = char_end
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
        self.meta = meta

    def __len__(self):
        return sum(map(len, self.words))

    def __eq__(self, other):
        try:
            return (
                self.sentence == other.sentence
                and self.char_start == other.char_start
                and self.char_end == other.char_end
                and self.expander_key == other.expander_key
                and self.position == other.position
            )
        except AttributeError:
            return False

    def __ne__(self, other):
        try:
            return (
                self.sentence != other.sentence
                or self.char_start != other.char_start
                or self.char_end != other.char_end
                or self.expander_key != other.expander_key
                or self.position != other.position
            )
        except AttributeError:
            return True

    def __hash__(self):
        return (
            hash(self.sentence)
            + hash(self.char_start)
            + hash(self.char_end)
            + hash(self.expander_key)
            + hash(self.position)
        )

    def get_stable_id(self):
        """
        Return a stable id.

        :rtype: string
        """
        doc_id, _, parent_doc_char_start, _ = split_stable_id(self.sentence.stable_id)
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

    def _get_table(self):
        return ImplicitSpanMention

    def _get_polymorphic_identity(self):
        return "implicit_span_mention"

    def _get_insert_args(self):
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

    def get_attrib_tokens(self, a="words"):
        """Get the tokens of sentence attribute *a*.

        Intuitively, like calling::

            implicit_span.a


        :param a: The attribute to get tokens for.
        :type a: str
        :return: The tokens of sentence attribute defined by *a* for the span.
        :rtype: list
        """
        return self.__getattribute__(a)

    def get_attrib_span(self, a, sep=" "):
        """Get the span of sentence attribute *a*.

        Intuitively, like calling::

            sep.join(implicit_span.a)

        :param a: The attribute to get a span for.
        :type a: str
        :param sep: The separator to use for the join.
        :type sep: str
        :return: The joined tokens, or text if a="words".
        :rtype: str
        """
        if a == "words":
            return self.text
        else:
            return sep.join(self.get_attrib_tokens(a))

    def __getitem__(self, key):
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

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"("
            f'"{self.get_span()}", '
            f"sentence={self.sentence.id}, "
            f"words=[{self.get_word_start_index()},{self.get_word_end_index()}], "
            f"position=[{self.position}]"
            f")"
        )

    def _get_instance(self, **kwargs):
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
    sentence = relationship("Context", foreign_keys=sentence_id)

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

    def _get_instance(self, **kwargs):
        return ImplicitSpanMention(**kwargs)

    # We redefine these to use default semantics, overriding the operators
    # inherited from TemporarySpan
    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other

    def __hash__(self):
        return id(self)
