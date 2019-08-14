from builtins import object
from typing import Any, Dict

from sqlalchemy import Column, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import backref, relationship

from fonduer.parser.models.context import Context

INT_ARRAY_TYPE = postgresql.ARRAY(Integer)
STR_ARRAY_TYPE = postgresql.ARRAY(String)


class SentenceMixin(object):
    """A sentence Context in a Document."""

    def is_lingual(self) -> bool:
        return False

    def is_visual(self) -> bool:
        return False

    def is_tabular(self) -> bool:
        return False

    def is_structural(self) -> bool:
        return False


class LingualMixin(object):
    """A collection of lingual attributes."""

    @declared_attr
    def lemmas(cls) -> Column:
        """A list of the lemmas for each word in a ``Sentence``."""
        return Column(STR_ARRAY_TYPE)

    @declared_attr
    def pos_tags(cls) -> Column:
        """A list of POS tags for each word in a ``Sentence``."""
        return Column(STR_ARRAY_TYPE)

    @declared_attr
    def ner_tags(cls) -> Column:
        """A list of NER tags for each word in a ``Sentence``."""
        return Column(STR_ARRAY_TYPE)

    @declared_attr
    def dep_parents(cls) -> Column:
        """A list of the dependency parents for each word in a ``Sentence``."""
        return Column(INT_ARRAY_TYPE)

    @declared_attr
    def dep_labels(cls) -> Column:
        """A list of dependency labels for each word in a ``Sentence``."""
        return Column(STR_ARRAY_TYPE)

    def is_lingual(self) -> bool:
        """Whether or not the ``Sentence`` contains NLP information.

        :rtype: bool
        """
        return self.lemmas is not None


class TabularMixin(object):
    """A collection of tabular attributes."""

    @declared_attr
    def table_id(cls) -> Column:
        """The id of the parent ``Table``, if any."""
        return Column("table_id", ForeignKey("table.id"))

    @declared_attr
    def table(cls) -> relationship:
        """The parent ``Table``, if any."""
        return relationship(
            "Table",
            backref=backref("sentences", cascade="all, delete-orphan"),
            foreign_keys=lambda: cls.table_id,
        )

    @declared_attr
    def cell_id(cls) -> Column:
        """The id of the parent ``Cell``, if any."""
        return Column("cell_id", ForeignKey("cell.id"))

    @declared_attr
    def cell(cls) -> relationship:
        """The parent ``Cell``, if any."""
        return relationship(
            "Cell",
            backref=backref("sentences", cascade="all, delete-orphan"),
            foreign_keys=lambda: cls.cell_id,
        )

    @declared_attr
    def row_start(cls) -> Column:
        """The ``row_start`` of the parent ``Cell``, if any."""
        return Column(Integer)

    @declared_attr
    def row_end(cls) -> Column:
        """The ``row_end`` of the parent ``Cell``, if any."""
        return Column(Integer)

    @declared_attr
    def col_start(cls) -> Column:
        """The ``col_start`` of the parent ``Cell``, if any."""
        return Column(Integer)

    @declared_attr
    def col_end(cls) -> Column:
        """The ``col_end`` of the parent ``Cell``, if any."""
        return Column(Integer)

    def is_tabular(self) -> bool:
        """Whether or not the ``Sentence`` contains tabular information.

        :rtype: bool
        """
        return self.table is not None

    def is_cellular(self) -> bool:
        """Whether or not the ``Sentence`` contains information about its table cell.

        :rtype: bool
        """
        return self.cell is not None


class VisualMixin(object):
    """A collection of visual attributes."""

    @declared_attr
    def page(cls) -> Column:
        """A list of the page index of each word in the ``Sentence``.

        Page indexes start at 0.
        """
        return Column(INT_ARRAY_TYPE)

    @declared_attr
    def top(cls) -> Column:
        """A list of each word's TOP bounding box coordinate in the ``Sentence``."""
        return Column(INT_ARRAY_TYPE)

    @declared_attr
    def left(cls) -> Column:
        """A list of each word's LEFT bounding box coordinate in the ``Sentence``."""
        return Column(INT_ARRAY_TYPE)

    @declared_attr
    def bottom(cls) -> Column:
        """A list of each word's BOTTOM bounding box coordinate in the ``Sentence``."""
        return Column(INT_ARRAY_TYPE)

    @declared_attr
    def right(cls) -> Column:
        """A list of each word's RIGHT bounding box coordinate in the ``Sentence``."""
        return Column(INT_ARRAY_TYPE)

    def is_visual(self) -> bool:
        """Whether or not the ``Sentence`` contains visual information.

        :rtype: bool
        """
        return self.page is not None and self.page[0] is not None


class StructuralMixin(object):
    """A collection of structural attributes."""

    @declared_attr
    def xpath(cls) -> Column:
        """The HTML XPATH to the ``Sentence``."""
        return Column(String)

    @declared_attr
    def html_tag(cls) -> Column:
        """The HTML tag of the element containing the ``Sentence``."""
        return Column(String)

    #: The HTML attributes of the element the ``Sentence`` is found in.
    @declared_attr
    def html_attrs(cls) -> Column:
        """A list of the html attributes of the element containing the ``Sentence``."""
        return Column(STR_ARRAY_TYPE)

    def is_structural(self) -> bool:
        """Whether or not the ``Sentence`` contains structural information.

        :rtype: bool
        """
        return self.html_tag is not None


# SentenceMixin must come last in arguments to not ovewrite is_* methods
# class Sentence(Context, StructuralMixin, SentenceMixin): # Memex variant
class Sentence(
    Context, TabularMixin, LingualMixin, VisualMixin, StructuralMixin, SentenceMixin
):
    """A Sentence subclass with Lingual, Tabular, Visual, and HTML attributes.

    .. note:: Unlike other data models, there is no HTML element corresponding to
        ``Sentence``. One ``Paragraph`` comprises one or more of ``Sentence``, but how a
        ``Paragraph`` is split depends on which NLP parser (e.g., spaCy) is used.
    """

    __tablename__ = "sentence"

    #: The unique id for the ``Sentence``.
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)

    #: The position of the ``Sentence`` in the ``Document``.
    position = Column(Integer, nullable=False)  # unique sentence number per document

    #: The name of a ``Sentence``.
    name = Column(String, unique=False, nullable=True)

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

    def __repr__(self) -> str:
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
                f"Sentence ("
                f"Doc: '{self.document.name}', "
                f"Table: {self.table.position}, "
                f"Row: {rows}, "
                f"Col: {cols}, "
                f"Index: {self.position}, "
                f"Text: '{self.text}'"
                f")"
            )

        else:
            return (
                f"Sentence ("
                f"Doc: '{self.document.name}', "
                f"Sec: {self.section.position}, "
                f"Par: {self.paragraph.position}, "
                f"Idx: {self.position}, "
                f"Text: '{self.text}'"
                f")"
            )

    def _asdict(self) -> Dict[str, Any]:
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

    def __gt__(self, other: "Sentence") -> bool:
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()
