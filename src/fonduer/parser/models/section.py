"""Fonduer section context model."""
from sqlalchemy import Column, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import backref, relationship

from fonduer.parser.models.context import Context


class Section(Context):
    """A Section Context in a Document.

    .. note:: As of v0.6.2, each document simply has a single Section.
        Specifically, ``<html>`` and ``<section>`` tags turn into ``Section``.
        Future parsing improvements can add better section recognition, such as the
        sections of an academic paper.
    """

    __tablename__ = "section"

    #: The unique id of the ``Section``.
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)

    #: The name of a ``Section``.
    name = Column(String, unique=False, nullable=True)

    #: The position of the ``Section`` in a ``Document``.
    position = Column(Integer, nullable=False)

    #: The id of the parent ``Document``.
    document_id = Column(Integer, ForeignKey("document.id", ondelete="CASCADE"))
    #: The parent ``Document``.
    document = relationship(
        "Document",
        backref=backref("sections", order_by=position, cascade="all, delete-orphan"),
        foreign_keys=document_id,
    )

    __mapper_args__ = {"polymorphic_identity": "section"}

    __table_args__ = (UniqueConstraint(document_id, position),)

    def __repr__(self) -> str:
        """Represent the context as a string."""
        return f"Section(Doc: {self.document.name}, Pos: {self.position})"

    def __gt__(self, other: "Section") -> bool:
        """Check if the context is greater than another context."""
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()
