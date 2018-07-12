from sqlalchemy import Column, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import backref, relationship
from sqlalchemy.sql import text
from sqlalchemy.types import PickleType

from fonduer.candidates.models.span import TemporarySpan
from fonduer.parser.models.context import Context, split_stable_id


class TemporaryImplicitSpan(TemporarySpan):
    """The TemporaryContext version of ImplicitSpan"""

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
        super(TemporarySpan, self).__init__()
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
        doc_id, _, parent_doc_char_start, _ = split_stable_id(self.sentence.stable_id)
        # return (construct_stable_id(self.sentence, self._get_polymorphic_identity(), self.char_start, self.char_end)
        #     + ':%s:%s' % (self.expander_key, self.position))
        return "%s::%s:%s:%s:%s:%s" % (
            self.sentence.document.name,
            self._get_polymorphic_identity(),
            parent_doc_char_start + self.char_start,
            parent_doc_char_start + self.char_end,
            self.expander_key,
            self.position,
        )

    def _get_table_name(self):
        return "implicit_span"

    def _get_polymorphic_identity(self):
        return "implicit_span"

    def _get_insert_query(self):
        return """INSERT INTO implicit_span VALUES(
            :id,
            :sentence_id,
            :char_start,
            :char_end,
            :expander_key,
            :position,
            :text,
            :words,
            :lemmas,
            :pos_tags,
            :ner_tags,
            :dep_parents,
            :dep_labels,
            :page,
            :top,
            :left,
            :bottom,
            :right,
            :meta)"""

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
        """Get the tokens of sentence attribute *a* over the range defined by word_offset, n"""
        return self.__getattribute__(a)

    def get_attrib_span(self, a, sep=" "):
        """Get the span of sentence attribute *a* over the range defined by word_offset, n"""
        if a == "words":
            return self.text
        else:
            return sep.join(self.get_attrib_tokens(a))

    def __getitem__(self, key):
        """Slice operation returns a new candidate sliced according to **char index**.

        Note that the slicing is w.r.t. the candidate range (not the abs. sentence char indexing)
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
        return '{}("{}", sentence={}, words=[{},{}], position=[{}])'.format(
            self.__class__.__name__,
            self.get_span(),
            self.sentence.id,
            self.get_word_start(),
            self.get_word_end(),
            self.position,
        )

    def _get_instance(self, **kwargs):
        return TemporaryImplicitSpan(**kwargs)


class ImplicitSpan(Context, TemporaryImplicitSpan):
    """A span of characters that may not have appeared verbatim in the source text.

    It is identified by Context id, character-index start and end (inclusive),
    as well as a key representing what 'expander' function drew the ImplicitSpan
    from an  existing Span, and a position (where position=0 corresponds to the
    first ImplicitSpan produced from the expander function).

    The character-index start and end point to the segment of text that was
    expanded to produce the ImplicitSpan.
    """

    __tablename__ = "implicit_span"
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)
    sentence_id = Column(
        Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True
    )
    char_start = Column(Integer, nullable=False)
    char_end = Column(Integer, nullable=False)
    expander_key = Column(String, nullable=False)
    position = Column(Integer, nullable=False)
    text = Column(String)
    words = Column(postgresql.ARRAY(String), nullable=False)
    lemmas = Column(postgresql.ARRAY(String))
    pos_tags = Column(postgresql.ARRAY(String))
    ner_tags = Column(postgresql.ARRAY(String))
    dep_parents = Column(postgresql.ARRAY(Integer))
    dep_labels = Column(postgresql.ARRAY(String))
    page = Column(postgresql.ARRAY(Integer))
    top = Column(postgresql.ARRAY(Integer))
    left = Column(postgresql.ARRAY(Integer))
    bottom = Column(postgresql.ARRAY(Integer))
    right = Column(postgresql.ARRAY(Integer))
    meta = Column(PickleType)

    __table_args__ = (
        UniqueConstraint(sentence_id, char_start, char_end, expander_key, position),
    )

    __mapper_args__ = {
        "polymorphic_identity": "implicit_span",
        "inherit_condition": (id == Context.id),
    }

    sentence = relationship(
        "Context",
        backref=backref("implicit_spans", cascade="all, delete-orphan"),
        foreign_keys=sentence_id,
    )

    def _get_instance(self, **kwargs):
        return ImplicitSpan(**kwargs)

    # We redefine these to use default semantics, overriding the operators inherited from TemporarySpan
    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other

    def __hash__(self):
        return id(self)
