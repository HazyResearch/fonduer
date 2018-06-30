from fonduer.parser.models.context import Context
from sqlalchemy import Column, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import backref, relationship


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

    __mapper_args__ = {"polymorphic_identity": "table"}

    __table_args__ = (UniqueConstraint(document_id, position),)

    def __repr__(self):
        return "Table(Doc: {}, Position: {})".format(self.document.name, self.position)

    def __gt__(self, other):
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()
