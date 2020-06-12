"""Fonduer caption context model."""
from sqlalchemy import Column, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import backref, relationship

from fonduer.parser.models.context import Context


class Caption(Context):
    """A Caption Context in a Document.

    Used to represent figure or table captions in a document.

    .. note:: As of v0.6.2, ``<caption>`` and ``<figcaption>`` tags turn into
        ``Caption``.
    """

    __tablename__ = "caption"

    #: The unique id the ``Caption``.
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)

    #: The position of the ``Caption`` in the ``Document``.
    position = Column(Integer, nullable=False)

    #: The name of a ``Caption``.
    name = Column(String, unique=False, nullable=True)

    #: The id of the parent ``Document``.
    document_id = Column(Integer, ForeignKey("document.id"))
    #: The parent ``Document``.
    document = relationship(
        "Document",
        backref=backref("captions", order_by=position, cascade="all, delete-orphan"),
        foreign_keys=document_id,
    )

    #: The id of the parent ``Table``, if any.
    table_id = Column(Integer, ForeignKey("table.id"))
    #: The parent ``Table``, if any.
    table = relationship(
        "Table",
        backref=backref("captions", order_by=position, cascade="all, delete-orphan"),
        foreign_keys=table_id,
    )

    #: The id of the parent ``Figure``, if any.
    figure_id = Column(Integer, ForeignKey("figure.id"))
    #: The parent ``Figure``, if any.
    figure = relationship(
        "Figure",
        backref=backref("captions", order_by=position, cascade="all, delete-orphan"),
        foreign_keys=figure_id,
    )

    __mapper_args__ = {"polymorphic_identity": "caption"}

    __table_args__ = (UniqueConstraint(document_id, table_id, figure_id, position),)

    def __repr__(self) -> str:
        """Represent the context as a string."""
        if self.figure:
            return (
                f"Caption("
                f"Doc: {self.document.name}, "
                f"Figure: {self.figure.position}, "
                f"Pos: {self.position}"
                f")"
            )
        elif self.table:
            return (
                f"Caption("
                f"Doc: {self.document.name}, "
                f"Table: {self.table.position}, "
                f"Pos: {self.position}"
                f")"
            )
        else:
            raise NotImplementedError(
                "Caption must be associated with Figure or Table."
            )

    def __gt__(self, other: "Caption") -> bool:
        """Check if the context is greater than another context."""
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()
