from sqlalchemy import Column, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import backref, relationship

from fonduer.parser.models.context import Context


class Caption(Context):
    """A Caption Context in a Document.

    Used to represent figure or table captions in a document.
    """

    __tablename__ = "caption"

    #: The unique id the ``Caption``.
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)

    #: The position of the ``Caption`` in the ``Document``.
    position = Column(Integer, nullable=False)

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

    def __repr__(self):
        if self.figure:
            return "Caption(Doc: {}, Figure: {}, Pos: {})".format(
                self.document.name, self.figure.position, self.position
            )
        elif self.table:
            return "Caption(Doc: {}, Table: {}, Pos: {})".format(
                self.document.name, self.table.position, self.position
            )
        else:
            raise NotImplementedError(
                "Caption must be associated with Figure or Table."
            )

    def __gt__(self, other):
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()
