from sqlalchemy import Column, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import backref, relationship

from fonduer.parser.models.context import Context


class Figure(Context):
    """A figure Context in a Document."""

    __tablename__ = "figure"
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)
    document_id = Column(Integer, ForeignKey("document.id", ondelete="CASCADE"))
    position = Column(Integer, nullable=False)
    document = relationship(
        "Document",
        backref=backref("figures", order_by=position, cascade="all, delete-orphan"),
        foreign_keys=document_id,
    )
    section_id = Column(Integer, ForeignKey("section.id"))
    section = relationship(
        "Section",
        backref=backref("figures", order_by=position, cascade="all, delete-orphan"),
        foreign_keys=section_id,
    )
    url = Column(String)

    __mapper_args__ = {"polymorphic_identity": "figure"}

    __table_args__ = (UniqueConstraint(document_id, position),)

    def __repr__(self):
        return "Figure(Doc: {}, Sec: {}, Pos: {}, Url: {})".format(
            self.document.name, self.section.position, self.position, self.url
        )

    def __gt__(self, other):
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()
