import logging
from builtins import object

from fonduer.meta import Meta
from sqlalchemy import Column, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import backref, relationship
from sqlalchemy.sql import select, text

# Grab pointer to global metadata
_meta = Meta.init()

INT_ARRAY_TYPE = postgresql.ARRAY(Integer)
STR_ARRAY_TYPE = postgresql.ARRAY(String)

logger = logging.getLogger(__name__)


class Context(_meta.Base):
    """
    A piece of content from which Candidates are composed.
    """

    __tablename__ = "context"
    id = Column(Integer, primary_key=True)
    type = Column(String, nullable=False)
    stable_id = Column(String, unique=True, nullable=False)

    __mapper_args__ = {"polymorphic_identity": "context", "polymorphic_on": type}

    def get_parent(self):
        raise NotImplementedError()

    def get_children(self):
        raise NotImplementedError()

    def get_sentence_generator(self):
        raise NotImplementedError()


class TemporaryContext(object):
    """
    A context which does not incur the overhead of a proper ORM-based Context
    object. The TemporaryContext class is specifically for the candidate
    extraction process, during which a CandidateSpace object will generate many
    TemporaryContexts, which will then be filtered by Matchers prior to
    materialization of Candidates and constituent Context objects.

    Every Context object has a corresponding TemporaryContext object from which
    it inherits.

    A TemporaryContext must have specified equality / set membership semantics,
    a stable_id for checking uniqueness against the database, and a promote()
    method which returns a corresponding Context object.
    """

    def __init__(self):
        self.id = None

    def load_id_or_insert(self, session):
        if self.id is None:
            stable_id = self.get_stable_id()
            id = session.execute(
                select([Context.id]).where(Context.stable_id == stable_id)
            ).first()
            if id is None:
                self.id = session.execute(
                    Context.__table__.insert(),
                    {"type": self._get_table_name(), "stable_id": stable_id},
                ).inserted_primary_key[0]
                insert_args = self._get_insert_args()
                insert_args["id"] = self.id
                for (key, val) in insert_args.items():
                    if isinstance(val, list):
                        insert_args[key] = val
                session.execute(text(self._get_insert_query()), insert_args)
            else:
                self.id = id[0]

    def __eq__(self, other):
        raise NotImplementedError()

    def __ne__(self, other):
        raise NotImplementedError()

    def __hash__(self):
        raise NotImplementedError()

    def _get_polymorphic_identity(self):
        raise NotImplementedError()

    def get_stable_id(self):
        raise NotImplementedError()

    def _get_table_name(self):
        raise NotImplementedError()

    def _get_insert_query(self):
        raise NotImplementedError()

    def _get_insert_args(self):
        raise NotImplementedError()


class PhraseMixin(object):
    """A phrase Context in a Document."""

    def is_lingual(self):
        return False

    def is_visual(self):
        return False

    def is_tabular(self):
        return False

    def is_structural(self):
        return False

    def __repr__(self):
        return "Phrase (Doc: {}, Index: {}, Text: {})".format(
            self.document.name, self.phrase_idx, self.text
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
        return "LingualPhrase (Doc: {}, Index: {}, Text: {})".format(
            self.document.name, self.phrase_idx, self.text
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
            backref=backref("phrases", cascade="all, delete-orphan"),
            foreign_keys=lambda: cls.table_id,
        )

    @declared_attr
    def cell_id(cls):
        return Column("cell_id", ForeignKey("cell.id"))

    @declared_attr
    def cell(cls):
        return relationship(
            "Cell",
            backref=backref("phrases", cascade="all, delete-orphan"),
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
        return "TabularPhrase (Doc: {}, Table: {}, Row: {}, Col: {}, Index: {}, Text: {})".format(
            self.document.name,
            (lambda: self.table).position,
            rows,
            cols,
            self.phrase_idx,
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
        return "VisualPhrase (Doc: {}, Page: {}, (T,B,L,R): ({},{},{},{}), Text: {})".format(
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
        return "StructuralPhrase (Doc: {}, Tag: {}, Text: {})".format(
            self.document.name, self.html_tag, self.text
        )


# PhraseMixin must come last in arguments to not ovewrite is_* methods
# class Phrase(Context, StructuralMixin, PhraseMixin): # Memex variant
class Phrase(
    Context, TabularMixin, LingualMixin, VisualMixin, StructuralMixin, PhraseMixin
):
    """A Phrase subclass with Lingual, Tabular, Visual, and HTML attributes."""

    __tablename__ = "phrase"
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)
    document_id = Column(Integer, ForeignKey("document.id"))
    document = relationship(
        "Document",
        backref=backref("phrases", cascade="all, delete-orphan"),
        foreign_keys=document_id,
    )
    phrase_num = Column(Integer, nullable=False)  # unique Phrase number per document
    text = Column(Text, nullable=False)
    words = Column(STR_ARRAY_TYPE)
    char_offsets = Column(INT_ARRAY_TYPE)
    entity_cids = Column(STR_ARRAY_TYPE)
    entity_types = Column(STR_ARRAY_TYPE)
    abs_char_offsets = Column(INT_ARRAY_TYPE)

    __mapper_args__ = {"polymorphic_identity": "phrase"}

    __table_args__ = (UniqueConstraint(document_id, phrase_num),)

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
            return "Phrase (testDoc: '{}', Table: {}, Row: {}, Col: {}, Index: {}, Text: '{}')".format(
                self.document.name,
                self.table.position,
                rows,
                cols,
                self.position,
                self.text,
            )
        else:
            return "Phrase (Doc: '{}', Index: {}, Text: '{}')".format(
                self.document.name, self.phrase_num, self.text
            )

    def _asdict(self):
        return {
            # base
            "id": self.id,
            # 'document': self.document,
            "phrase_num": self.phrase_num,
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
            "position": self.position,
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


def construct_stable_id(
    parent_context,
    polymorphic_type,
    relative_char_offset_start,
    relative_char_offset_end,
):
    """Contruct a stable ID for a Context given its parent and its character offsets relative to the parent"""
    doc_id, _, parent_doc_char_start, _ = split_stable_id(parent_context.stable_id)
    start = parent_doc_char_start + relative_char_offset_start
    end = parent_doc_char_start + relative_char_offset_end
    return "%s::%s:%s:%s" % (doc_id, polymorphic_type, start, end)


def split_stable_id(stable_id):
    """Split stable id, returning:

        * Document (root) stable ID
        * Context polymorphic type
        * Character offset start, end *relative to document start*

    Returns tuple of four values.
    """
    split1 = stable_id.split("::")
    if len(split1) == 2:
        split2 = split1[1].split(":")
        if len(split2) == 3:
            return split1[0], split2[0], int(split2[1]), int(split2[2])
    raise ValueError("Malformed stable_id:", stable_id)
