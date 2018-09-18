from builtins import object

from sqlalchemy import Column, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import backref, relationship

from fonduer.parser.models.context import Context

INT_ARRAY_TYPE = postgresql.ARRAY(Integer)
STR_ARRAY_TYPE = postgresql.ARRAY(String)


class SentenceMixin(object):
    """A sentence Context in a Document."""

    def is_lingual(self):
        return False

    def is_visual(self):
        return False

    def is_tabular(self):
        return False

    def is_structural(self):
        return False

    def __repr__(self):
        return "Sentence (Doc: {}, Index: {}, Text: {})".format(
            self.document.name, self.sentence_idx, self.text
        )


class LingualMixin(object):
    """A collection of lingual attributes."""

    @declared_attr
    def lemmas(cls):
        """A list of the lemmas for each word in a ``Sentence``."""
        return Column(STR_ARRAY_TYPE)

    @declared_attr
    def pos_tags(cls):
        """A list of POS tags for each word in a ``Sentence``."""
        return Column(STR_ARRAY_TYPE)

    @declared_attr
    def ner_tags(cls):
        """A list of NER tags for each word in a ``Sentence``."""
        return Column(STR_ARRAY_TYPE)

    @declared_attr
    def dep_parents(cls):
        """A list of the dependency parents for each word in a ``Sentence``."""
        return Column(INT_ARRAY_TYPE)

    @declared_attr
    def dep_labels(cls):
        """A list of dependency labels for each word in a ``Sentence``."""
        return Column(STR_ARRAY_TYPE)

    def is_lingual(self):
        """Whether or not the ``Sentence`` contains NLP information.

        :rtype: bool
        """
        return self.lemmas is not None

    def __repr__(self):
        return "LingualSentence (Doc: {}, Index: {}, Text: {})".format(
            self.document.name, self.sentence_idx, self.text
        )


class TabularMixin(object):
    """A collection of tabular attributes."""

    @declared_attr
    def table_id(cls):
        """The id of the parent ``Table``, if any."""
        return Column("table_id", ForeignKey("table.id"))

    @declared_attr
    def table(cls):
        """The parent ``Table``, if any."""
        return relationship(
            "Table",
            backref=backref("sentences", cascade="all, delete-orphan"),
            foreign_keys=lambda: cls.table_id,
        )

    @declared_attr
    def cell_id(cls):
        """The id of the parent ``Cell``, if any."""
        return Column("cell_id", ForeignKey("cell.id"))

    @declared_attr
    def cell(cls):
        """The parent ``Cell``, if any."""
        return relationship(
            "Cell",
            backref=backref("sentences", cascade="all, delete-orphan"),
            foreign_keys=lambda: cls.cell_id,
        )

    @declared_attr
    def row_start(cls):
        """The ``row_start`` of the parent ``Cell``, if any."""
        return Column(Integer)

    @declared_attr
    def row_end(cls):
        """The ``row_end`` of the parent ``Cell``, if any."""
        return Column(Integer)

    @declared_attr
    def col_start(cls):
        """The ``col_start`` of the parent ``Cell``, if any."""
        return Column(Integer)

    @declared_attr
    def col_end(cls):
        """The ``col_end`` of the parent ``Cell``, if any."""
        return Column(Integer)

    def is_tabular(self):
        """Whether or not the ``Sentence`` contains tabular information.

        :rtype: bool
        """
        return self.table is not None

    def is_cellular(self):
        """Whether or not the ``Sentence`` contains information about its table cell.

        :rtype: bool
        """
        return self.cell is not None

    def __repr__(self):
        rows = (
            tuple([self.row_start, self.row_end])
            if self.row_start != self.row_end
            else self.row_start
        )
        cols = (
            tuple([self.col_start, self.col_end])
            if self.col_start != self.col_end
            else self.col_start
        )
        return (
            "TabularSentence (Doc: {}, "
            + "Table: {}, Row: {}, Col: {}, Index: {}, Text: {})"
        ).format(
            self.document.name,
            (lambda: self.table).position,
            rows,
            cols,
            self.sentence_idx,
            self.text,
        )


class VisualMixin(object):
    """A collection of visual attributes."""

    @declared_attr
    def page(cls):
        """A list of the page index of each word in the ``Sentence``.

        Page indexes start at 0.
        """
        return Column(INT_ARRAY_TYPE)

    @declared_attr
    def top(cls):
        """A list of each word's TOP bounding box coordinate in the ``Sentence``."""
        return Column(INT_ARRAY_TYPE)

    @declared_attr
    def left(cls):
        """A list of each word's LEFT bounding box coordinate in the ``Sentence``."""
        return Column(INT_ARRAY_TYPE)

    @declared_attr
    def bottom(cls):
        """A list of each word's BOTTOM bounding box coordinate in the ``Sentence``."""
        return Column(INT_ARRAY_TYPE)

    @declared_attr
    def right(cls):
        """A list of each word's RIGHT bounding box coordinate in the ``Sentence``."""
        return Column(INT_ARRAY_TYPE)

    def is_visual(self):
        """Whether or not the ``Sentence`` contains visual information.

        :rtype: bool
        """
        return self.page is not None and self.page[0] is not None

    def __repr__(self):
        return (
            "VisualSentence (Doc: {}, Page: {}, (T,B,L,R): ({},{},{},{}), Text: {})"
        ).format(
            self.document.name,
            self.page,
            self.top,
            self.bottom,
            self.left,
            self.right,
            self.text,
        )


class StructuralMixin(object):
    """A collection of structural attributes."""

    @declared_attr
    def xpath(cls):
        """The HTML XPATH to the ``Sentence``."""
        return Column(String)

    @declared_attr
    def html_tag(cls):
        """The HTML tag of the element containing the ``Sentence``."""
        return Column(String)

    #: The HTML attributes of the element the ``Sentence`` is found in.
    @declared_attr
    def html_attrs(cls):
        """A list of the html attributes of the element containing the ``Sentence``."""
        return Column(STR_ARRAY_TYPE)

    def is_structural(self):
        """Whether or not the ``Sentence`` contains structural information.

        :rtype: bool
        """
        return self.html_tag is not None

    def __repr__(self):
        return "StructuralSentence (Doc: {}, Tag: {}, Text: {})".format(
            self.document.name, self.html_tag, self.text
        )


# SentenceMixin must come last in arguments to not ovewrite is_* methods
# class Sentence(Context, StructuralMixin, SentenceMixin): # Memex variant
class Sentence(
    Context, TabularMixin, LingualMixin, VisualMixin, StructuralMixin, SentenceMixin
):
    """A Sentence subclass with Lingual, Tabular, Visual, and HTML attributes."""

    __tablename__ = "sentence"

    #: The unique id for the ``Sentence``.
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)

    #: The position of the ``Sentence`` in the ``Document``.
    position = Column(Integer, nullable=False)  # unique sentence number per document

    #: The id of the parent ``Document``.
    document_id = Column(Integer, ForeignKey("document.id"))

    #: The the parent ``Document``.
    document = relationship(
        "Document",
        backref=backref("sentences", cascade="all, delete-orphan"),
        foreign_keys=document_id,
    )

    #: The id of the parent ``Section``.
    section_id = Column(Integer, ForeignKey("section.id"))
    #: The parent ``Section``.
    section = relationship(
        "Section",
        backref=backref("sentences", cascade="all, delete-orphan"),
        foreign_keys=section_id,
    )

    #: The id of the parent ``Paragraph``.
    paragraph_id = Column(Integer, ForeignKey("paragraph.id"))
    #: The parent ``Paragraph``.
    paragraph = relationship(
        "Paragraph",
        backref=backref("sentences", cascade="all, delete-orphan"),
        foreign_keys=paragraph_id,
    )

    #: The full text of the ``Sentence``.
    text = Column(Text, nullable=False)

    #: A list of the words in a ``Sentence``.
    words = Column(STR_ARRAY_TYPE)

    #: A list of the character offsets of each word in a ``Sentence``, with
    #: respect to the start of the sentence.
    char_offsets = Column(INT_ARRAY_TYPE)

    #: A list of the character offsets of each word in a ``Sentence``, with
    #: respect to the entire document.
    abs_char_offsets = Column(INT_ARRAY_TYPE)

    __mapper_args__ = {"polymorphic_identity": "sentence"}

    __table_args__ = (UniqueConstraint(document_id, position),)

    def __repr__(self):
        if self.is_tabular():
            rows = (
                tuple([self.row_start, self.row_end])
                if self.row_start != self.row_end
                else self.row_start
            )
            cols = (
                tuple([self.col_start, self.col_end])
                if self.col_start != self.col_end
                else self.col_start
            )
            return (
                "Sentence (Doc: '{}', "
                + "Table: {}, Row: {}, Col: {}, Index: {}, Text: '{}')"
            ).format(
                self.document.name,
                self.table.position,
                rows,
                cols,
                self.position,
                self.text,
            )

        else:
            return (
                "Sentence (Doc: '{}', Sec: {}, Par: {}, Idx: {}, Text: '{}')"
            ).format(
                self.document.name,
                self.section.position,
                self.paragraph.position,
                self.position,
                self.text,
            )

    def _asdict(self):
        return {
            # base
            "id": self.id,
            # 'document': self.document,
            "position": self.position,
            "text": self.text,
            # tabular
            # 'table': self.table,
            # 'cell': self.cell,
            "row_start": self.row_start,
            "row_end": self.row_end,
            "col_start": self.col_start,
            "col_end": self.col_end,
            # lingual
            "words": self.words,
            "char_offsets": self.char_offsets,
            "lemmas": self.lemmas,
            "pos_tags": self.pos_tags,
            "ner_tags": self.ner_tags,
            "dep_parents": self.dep_parents,
            "dep_labels": self.dep_labels,
            # visual
            "page": self.page,
            "top": self.top,
            "bottom": self.bottom,
            "left": self.left,
            "right": self.right,
        }

    def __gt__(self, other):
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()
