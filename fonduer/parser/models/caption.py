from sqlalchemy import Column, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import backref, relationship

from fonduer.parser.models.context import Context


class Caption(Context):
    """A caption Context in a Document."""

    __tablename__ = "caption"
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)
    position = Column(Integer, nullable=False)
    document_id = Column(Integer, ForeignKey("document.id"))
    document = relationship(
        "Document",
        backref=backref("captions", order_by=position, cascade="all, delete-orphan"),
        foreign_keys=document_id,
    )
    table_id = Column(Integer, ForeignKey("table.id"))
    table = relationship(
        "Table",
        backref=backref("captions", order_by=position, cascade="all, delete-orphan"),
        foreign_keys=table_id,
    )
    figure_id = Column(Integer, ForeignKey("figure.id"))
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
