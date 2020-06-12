"""Fonduer figure context model."""
from sqlalchemy import Column, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import backref, relationship

from fonduer.parser.models.context import Context


class Figure(Context):
    """A figure Context in a Document.

    Used to represent figures in a document.

    .. note:: As of v0.6.2, ``<img>`` and ``<figure>`` tags turn into ``Figure``.
    """

    __tablename__ = "figure"

    #: The unique id of the ``Figure``.
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)

    #: The position of the ``Figure`` in the ``Document``.
    position = Column(Integer, nullable=False)

    #: The name of a ``Figure``.
    name = Column(String, unique=False, nullable=True)

    #: The id of the parent ``Document``.
    document_id = Column(Integer, ForeignKey("document.id", ondelete="CASCADE"))
    #: The parent ``Document``.
    document = relationship(
        "Document",
        backref=backref("figures", order_by=position, cascade="all, delete-orphan"),
        foreign_keys=document_id,
    )

    #: The id of the parent ``Section``.
    section_id = Column(Integer, ForeignKey("section.id"))
    #: The parent ``Section``.
    section = relationship(
        "Section",
        backref=backref("figures", order_by=position, cascade="all, delete-orphan"),
        foreign_keys=section_id,
    )

    #: The id of the parent ``Cell``, if any.
    cell_id = Column(Integer, ForeignKey("cell.id"))
    #: The the parent ``Cell``, if any.
    cell = relationship(
        "Cell",
        backref=backref("figures", order_by=position, cascade="all, delete-orphan"),
        foreign_keys=cell_id,
    )

    #: The ``Figure``'s URL.
    url = Column(String)

    __mapper_args__ = {"polymorphic_identity": "figure"}

    __table_args__ = (UniqueConstraint(document_id, position),)

    def __repr__(self) -> str:
        """Represent the context as a string."""
        if self.cell:
            return (
                f"Figure("
                f"Doc: {self.document.name}, "
                f"Sec: {self.section.position}, "
                f"Cell: {self.cell.position}, "
                f"Pos: {self.position}, "
                f"Url: {self.url}"
                f")"
            )
        else:
            return (
                f"Figure("
                f"Doc: {self.document.name}, "
                f"Sec: {self.section.position}, "
                f"Pos: {self.position}, "
                f"Url: {self.url}"
                f")"
            )

    def __gt__(self, other: "Figure") -> bool:
        """Check if the context is greater than another context."""
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()
