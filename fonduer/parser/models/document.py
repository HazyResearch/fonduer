from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.types import PickleType

from fonduer.parser.models.context import Context


class Document(Context):
    """
    A root Context.
    """

    __tablename__ = "document"
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)
    name = Column(String, unique=True, nullable=False)
    text = Column(String)
    meta = Column(PickleType)

    __mapper_args__ = {"polymorphic_identity": "document"}

    def __repr__(self):
        return "Document " + str(self.name)

    def __gt__(self, other):
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()
