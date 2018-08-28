from sqlalchemy import Column, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import backref, relationship

from fonduer.parser.models.context import Context


class Section(Context):
    """A Section Context in a Document.

    Currently, each document simply has a single Section. Future parsing
    improvements can add better section recognition, such as the sections of an
    academic paper.
    """

    __tablename__ = "section"
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)
    document_id = Column(Integer, ForeignKey("document.id", ondelete="CASCADE"))
    position = Column(Integer, nullable=False)
    document = relationship(
        "Document",
        backref=backref("sections", order_by=position, cascade="all, delete-orphan"),
        foreign_keys=document_id,
    )

    __mapper_args__ = {"polymorphic_identity": "section"}

    __table_args__ = (UniqueConstraint(document_id, position),)

    def __repr__(self):
        return "Section(Doc: {}, Pos: {})".format(self.document.name, self.position)

    def __gt__(self, other):
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()
