"""Fonduer paragraph context model."""
from sqlalchemy import Column, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import backref, relationship

from fonduer.parser.models.context import Context


class Paragraph(Context):
    """A paragraph Context in a Document.

    Represents a grouping of adjacent sentences.

    .. note:: As of v0.6.2, a text content in two properties ``.text`` and ``.tail``
        turn into ``Paragraph``.
        See https://lxml.de/tutorial.html#elements-contain-text for details about
        ``.text`` and ``.tail`` properties.
    """

    __tablename__ = "paragraph"

    #: The unique id of the ``Paragraph``.
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)

    #: The position of the ``Paragraph`` in the ``Document``.
    position = Column(Integer, nullable=False)

    #: The name of a ``Paragraph``.
    name = Column(String, unique=False, nullable=True)

    #: The id of the parent ``Document``.
    document_id = Column(Integer, ForeignKey("document.id"))
    #: The parent ``Document``.
    document = relationship(
        "Document",
        backref=backref("paragraphs", order_by=position, cascade="all, delete-orphan"),
        foreign_keys=document_id,
    )

    #: The id of the parent ``Section``.
    section_id = Column(Integer, ForeignKey("section.id"))
    #: The parent ``Section``.
    section = relationship(
        "Section",
        backref=backref("paragraphs", cascade="all, delete-orphan"),
        foreign_keys=section_id,
    )

    #: The id of the parent ``Cell``, if any.
    cell_id = Column(Integer, ForeignKey("cell.id"))
    #: The parent ``Cell``, if any.
    cell = relationship(
        "Cell",
        backref=backref("paragraphs", cascade="all, delete-orphan"),
        foreign_keys=cell_id,
    )

    #: The id of the parent ``Caption``, if any.
    caption_id = Column(Integer, ForeignKey("caption.id"))
    #: The parent ``Caption``, if any.
    caption = relationship(
        "Caption",
        backref=backref("paragraphs", cascade="all, delete-orphan"),
        foreign_keys=caption_id,
    )

    __mapper_args__ = {"polymorphic_identity": "paragraph"}

    __table_args__ = (UniqueConstraint(document_id, position),)

    def __repr__(self) -> str:
        """Represent the context as a string."""
        if self.cell:
            return (
                f"Paragraph("
                f"Doc: {self.document.name}, "
                f"Sec: {self.section.position}, "
                f"Cell: {self.cell.position}, "
                f"Pos: {self.position}"
                f")"
            )
        elif self.caption:
            return (
                f"Paragraph("
                f"Doc: {self.document.name}, "
                f"Sec: {self.section.position}, "
                f"Caption: {self.caption.position}, "
                f"Pos: {self.position}"
                ")"
            )
        else:
            return (
                f"Paragraph("
                f"Doc: {self.document.name}, "
                f"Sec: {self.section.position}, "
                f"Pos: {self.position}"
                f")"
            )

    def __gt__(self, other: "Paragraph") -> bool:
        """Check if the context is greater than another context."""
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()
