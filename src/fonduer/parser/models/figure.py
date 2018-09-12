from sqlalchemy import Column, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import backref, relationship

from fonduer.parser.models.context import Context


class Figure(Context):
    """A figure Context in a Document.

    Used to represent figures in a document, such as ``<img>`` or ``<figure>``
    tags in HTML.
    """

    __tablename__ = "figure"

    #: The unique id of the ``Figure``.
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)

    #: The position of the ``Figure`` in the ``Document``.
    position = Column(Integer, nullable=False)

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

    def __repr__(self):
        if self.cell:
            return "Figure(Doc: {}, Sec: {}, Cell: {}, Pos: {}, Url: {})".format(
                self.document.name,
                self.section.position,
                self.cell.position,
                self.position,
                self.url,
            )
        else:
            return "Figure(Doc: {}, Sec: {}, Pos: {}, Url: {})".format(
                self.document.name, self.section.position, self.position, self.url
            )

    def __gt__(self, other):
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()
