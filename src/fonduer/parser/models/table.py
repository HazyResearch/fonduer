from sqlalchemy import Column, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import backref, relationship

from fonduer.parser.models.context import Context


class Table(Context):
    """A table Context in a Document."""

    __tablename__ = "table"
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)
    document_id = Column(Integer, ForeignKey("document.id"))
    position = Column(Integer, nullable=False)
    document = relationship(
        "Document",
        backref=backref("tables", order_by=position, cascade="all, delete-orphan"),
        foreign_keys=document_id,
    )
    section_id = Column(Integer, ForeignKey("section.id"))
    section = relationship(
        "Section",
        backref=backref("tables", order_by=position, cascade="all, delete-orphan"),
        foreign_keys=section_id,
    )

    __mapper_args__ = {"polymorphic_identity": "table"}

    __table_args__ = (UniqueConstraint(document_id, position),)

    def __repr__(self):
        return "Table(Doc: {}, Sec: {}, Position: {})".format(
            self.document.name, self.section.position, self.position
        )

    def __gt__(self, other):
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()


class Cell(Context):
    """A cell Context in a Document."""

    __tablename__ = "cell"
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)
    document_id = Column(Integer, ForeignKey("document.id"))
    table_id = Column(Integer, ForeignKey("table.id"))
    position = Column(Integer, nullable=False)
    document = relationship(
        "Document",
        backref=backref("cells", order_by=position, cascade="all, delete-orphan"),
        foreign_keys=document_id,
    )
    table = relationship(
        "Table",
        backref=backref("cells", order_by=position, cascade="all, delete-orphan"),
        foreign_keys=table_id,
    )
    row_start = Column(Integer)
    row_end = Column(Integer)
    col_start = Column(Integer)
    col_end = Column(Integer)
    html_tag = Column(Text)
    html_attrs = Column(postgresql.ARRAY(String))

    __mapper_args__ = {"polymorphic_identity": "cell"}

    __table_args__ = (UniqueConstraint(document_id, table_id, position),)

    def __repr__(self):
        return "Cell(Doc: {}, Table: {}, Row: {}, Col: {}, Pos: {})".format(
            self.document.name,
            self.table.position,
            tuple({self.row_start, self.row_end}),
            tuple({self.col_start, self.col_end}),
            self.position,
        )

    def __gt__(self, other):
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()
