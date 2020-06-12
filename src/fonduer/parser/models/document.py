"""Fonduer document context model."""
from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.types import PickleType

from fonduer.parser.models.context import Context


class Document(Context):
    """A document Context.

    Represents all the information of a particular document.
    What becomes a document depends on which child class of ``DocPreprocessor`` is used.

    .. note:: As of v0.6.2, each file is one document when ``HTMLDocPreprocessor`` or
        ``TextDocPreprocessor`` is used, each line in the input file is treated as one
        document when ``CSVDocPreprocessor`` or ``TSVDocPreprocessor`` is used.
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

    def __repr__(self) -> str:
        """Represent the context as a string."""
        return f"Document {self.name}"

    def __gt__(self, other: "Document") -> bool:
        """Check if the context is greater than another context."""
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()
