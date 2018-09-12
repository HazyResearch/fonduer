from sqlalchemy import Column, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import backref, relationship

from fonduer.parser.models.context import Context


class Section(Context):
    """A Section Context in a Document.

    .. note:: Currently, each document simply has a single Section. Future
        parsing improvements can add better section recognition, such as the
        sections of an academic paper.
    """

    __tablename__ = "section"

    #: The unique id of the ``Section``.
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)

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

    def __repr__(self):
        return "Section(Doc: {}, Pos: {})".format(self.document.name, self.position)

    def __gt__(self, other):
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()
