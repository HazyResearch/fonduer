"""Fonduer table context model."""
from sqlalchemy import Column, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import backref, relationship

from fonduer.parser.models.context import Context


class Table(Context):
    """A Table Context in a Document.

    Used to represent tables found in a document.

    .. note:: As of v0.6.2, ``<table>`` tags turn into ``Table``.
    """

    __tablename__ = "table"

    #: The unique id of the ``Table``.
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)

    #: The position of the ``Table`` in the ``Document``.
    position = Column(Integer, nullable=False)

    #: The name of a ``Table``.
    name = Column(String, unique=False, nullable=True)

    #: The id of the parent ``Document``.
    document_id = Column(Integer, ForeignKey("document.id"))
    #: The parent ``Document``.
    document = relationship(
        "Document",
        backref=backref("tables", order_by=position, cascade="all, delete-orphan"),
        foreign_keys=document_id,
    )

    #: The id of the parent ``Section``.
    section_id = Column(Integer, ForeignKey("section.id"))
    #: The parent ``Section``.
    section = relationship(
        "Section",
        backref=backref("tables", order_by=position, cascade="all, delete-orphan"),
        foreign_keys=section_id,
    )

    __mapper_args__ = {"polymorphic_identity": "table"}

    __table_args__ = (UniqueConstraint(document_id, position),)

    def __repr__(self) -> str:
        """Represent the context as a string."""
        return (
            f"Table("
            f"Doc: {self.document.name}, "
            f"Sec: {self.section.position}, "
            f"Position: {self.position}"
            f")"
        )

    def __gt__(self, other: "Table") -> bool:
        """Check if the context is greater than another context."""
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()


class Cell(Context):
    """A cell Context in a Document.

    Used to represent the cells that comprise a table in a document.

    .. note:: As of v0.6.2, ``<th>`` and ``<td>`` tags turn into ``Cell``.
    """

    __tablename__ = "cell"

    #: The unique id of the ``Cell``.
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)

    #: The position of the ``Cell`` in the ``Table``.
    position = Column(Integer, nullable=False)

    #: The name of a ``Cell``.
    name = Column(String, unique=False, nullable=True)

    #: The id of the parent ``Table``.
    table_id = Column(Integer, ForeignKey("table.id"))
    #: The parent ``Table``.
    table = relationship(
        "Table",
        backref=backref("cells", order_by=position, cascade="all, delete-orphan"),
        foreign_keys=table_id,
    )

    #: The id of the parent ``Document``.
    document_id = Column(Integer, ForeignKey("document.id"))
    #: The parent ``Document``.
    document = relationship(
        "Document",
        backref=backref("cells", order_by=position, cascade="all, delete-orphan"),
        foreign_keys=document_id,
    )

    #: The start index of the row in the ``Table`` the ``Cell`` is in.
    row_start = Column(Integer)
    #: The end index of the row in the ``Table`` the ``Cell`` is in.
    row_end = Column(Integer)

    #: The start index of the column in the ``Table`` the ``Cell`` is in.
    col_start = Column(Integer)

    #: The end index of the column in the ``Table`` the ``Cell`` is in.
    col_end = Column(Integer)

    __mapper_args__ = {"polymorphic_identity": "cell"}

    __table_args__ = (UniqueConstraint(document_id, table_id, position),)

    def __repr__(self) -> str:
        """Represent the context as a string."""
        return (
            f"Cell(Doc: {self.document.name}, "
            f"Table: {self.table.position}, "
            f"Row: {tuple({self.row_start, self.row_end})}, "
            f"Col: {tuple({self.col_start, self.col_end})}, "
            f"Pos: {self.position})"
        )

    def __gt__(self, other: "Cell") -> bool:
        """Check if the context is greater than another context."""
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()
