from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.types import PickleType

from fonduer.parser.models.context import Context


class Document(Context):
    """A document Context.

    Represents all the information of a particular document.
    """

    __tablename__ = "document"

    #: The unique id of a ``Document``.
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)

    #: The filename of a ``Document``, without its extension (e.g., "BC818").
    name = Column(String, unique=True, nullable=False)

    #: The full text of the ``Document``.
    text = Column(String)

    #: Pickled metadata about a document extrated from a document preprocessor.
    meta = Column(PickleType)

    __mapper_args__ = {"polymorphic_identity": "document"}

    def __repr__(self):
        return "Document " + str(self.name)

    def __gt__(self, other):
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()
