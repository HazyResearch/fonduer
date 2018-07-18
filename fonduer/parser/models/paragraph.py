from sqlalchemy import Column, ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.orm import backref, relationship

from fonduer.parser.models.context import Context


class Paragraph(Context):
    """A paragraph Context in a Document."""

    __tablename__ = "paragraph"
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)
    document_id = Column(Integer, ForeignKey("document.id"))
    position = Column(Integer, nullable=False)
    document = relationship(
        "Document",
        backref=backref("paragraphs", order_by=position, cascade="all, delete-orphan"),
        foreign_keys=document_id,
    )
    section_id = Column(Integer, ForeignKey("section.id"))
    section = relationship(
        "Section",
        backref=backref("paragraphs", cascade="all, delete-orphan"),
        foreign_keys=section_id,
    )
    table_id = Column(Integer, ForeignKey("table.id"))
    table = relationship(
        "Table",
        backref=backref("paragraphs", cascade="all, delete-orphan"),
        foreign_keys=table_id,
    )
    cell_id = Column(Integer, ForeignKey("cell.id"))
    cell = relationship(
        "Cell",
        backref=backref("paragraphs", cascade="all, delete-orphan"),
        foreign_keys=cell_id,
    )

    __mapper_args__ = {"polymorphic_identity": "paragraph"}

    __table_args__ = (UniqueConstraint(document_id, position),)

    def __repr__(self):
        return "Paragraph(Doc: {}, Pos: {})".format(self.document.name, self.position)

    def __gt__(self, other):
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()
