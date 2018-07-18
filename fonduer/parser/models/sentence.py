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

    lemmas = Column(STR_ARRAY_TYPE)
    pos_tags = Column(STR_ARRAY_TYPE)
    ner_tags = Column(STR_ARRAY_TYPE)
    dep_parents = Column(INT_ARRAY_TYPE)
    dep_labels = Column(STR_ARRAY_TYPE)

    def is_lingual(self):
        return self.lemmas is not None

    def __repr__(self):
        return "LingualSentence (Doc: {}, Index: {}, Text: {})".format(
            self.document.name, self.sentence_idx, self.text
        )


class TabularMixin(object):
    """A collection of tabular attributes."""

    @declared_attr
    def table_id(cls):
        return Column("table_id", ForeignKey("table.id"))

    @declared_attr
    def table(cls):
        return relationship(
            "Table",
            backref=backref("sentences", cascade="all, delete-orphan"),
            foreign_keys=lambda: cls.table_id,
        )

    @declared_attr
    def cell_id(cls):
        return Column("cell_id", ForeignKey("cell.id"))

    @declared_attr
    def cell(cls):
        return relationship(
            "Cell",
            backref=backref("sentences", cascade="all, delete-orphan"),
            foreign_keys=lambda: cls.cell_id,
        )

    row_start = Column(Integer)
    row_end = Column(Integer)
    col_start = Column(Integer)
    col_end = Column(Integer)
    position = Column(Integer)

    def is_tabular(self):
        return self.table is not None

    def is_cellular(self):
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
        return "TabularSentence (Doc: {}, Table: {}, Row: {}, Col: {}, Index: {}, Text: {})".format(
            self.document.name,
            (lambda: self.table).position,
            rows,
            cols,
            self.sentence_idx,
            self.text,
        )


class VisualMixin(object):
    """A collection of visual attributes."""

    page = Column(INT_ARRAY_TYPE)
    top = Column(INT_ARRAY_TYPE)
    left = Column(INT_ARRAY_TYPE)
    bottom = Column(INT_ARRAY_TYPE)
    right = Column(INT_ARRAY_TYPE)

    def is_visual(self):
        return self.page is not None and self.page[0] is not None

    def __repr__(self):
        return "VisualSentence (Doc: {}, Page: {}, (T,B,L,R): ({},{},{},{}), Text: {})".format(
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

    xpath = Column(String)
    html_tag = Column(String)
    html_attrs = Column(STR_ARRAY_TYPE)

    def is_structural(self):
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
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)
    document_id = Column(Integer, ForeignKey("document.id"))
    document = relationship(
        "Document",
        backref=backref("sentences", cascade="all, delete-orphan"),
        foreign_keys=document_id,
    )
    section_id = Column(Integer, ForeignKey("section.id"))
    section = relationship(
        "Section",
        backref=backref("sentences", cascade="all, delete-orphan"),
        foreign_keys=section_id,
    )
    paragraph_id = Column(Integer, ForeignKey("paragraph.id"))
    paragraph = relationship(
        "Paragraph",
        backref=backref("sentences", cascade="all, delete-orphan"),
        foreign_keys=paragraph_id,
    )
    position = Column(Integer, nullable=False)  # unique sentence number per document
    text = Column(Text, nullable=False)
    words = Column(STR_ARRAY_TYPE)
    char_offsets = Column(INT_ARRAY_TYPE)
    entity_cids = Column(STR_ARRAY_TYPE)
    entity_types = Column(STR_ARRAY_TYPE)
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
            return "Sentence (Doc: '{}', Table: {}, Row: {}, Col: {}, Index: {}, Text: '{}')".format(
                self.document.name,
                self.table.position,
                rows,
                cols,
                self.position,
                self.text,
            )
        else:
            return "Sentence (Doc: '{}', Sec: {}, Par: {}, Index: {}, Text: '{}')".format(
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
            "entity_cids": self.entity_cids,
            "entity_types": self.entity_types,
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
