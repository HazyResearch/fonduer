from typing import Any, Dict, Type

from sqlalchemy import Column, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import relationship

from fonduer.candidates.models.temporary_context import TemporaryContext
from fonduer.parser.models import Document
from fonduer.parser.models.context import Context
from fonduer.parser.models.utils import construct_stable_id


class TemporaryDocumentMention(TemporaryContext):
    """The TemporaryContext version of DocumentMention."""

    def __init__(self, document: Document) -> None:
        super().__init__()
        self.document = document  # The document Context

    def __len__(self) -> int:
        return 1

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TemporaryDocumentMention):
            return NotImplemented
        return self.document == other.document

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, TemporaryDocumentMention):
            return NotImplemented
        return self.document != other.document

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, TemporaryDocumentMention):
            return NotImplemented
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()

    def __contains__(self, other: object) -> bool:
        if not isinstance(other, TemporaryDocumentMention):
            return NotImplemented
        return self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self.document)

    def get_stable_id(self) -> str:
        """
        Return a stable id.

        :rtype: string
        """
        return construct_stable_id(
            self.document, self._get_polymorphic_identity(), 0, 0
        )

    def _get_table(self) -> Type["DocumentMention"]:
        return DocumentMention

    def _get_polymorphic_identity(self) -> str:
        return "document_mention"

    def _get_insert_args(self) -> Dict[str, Any]:
        return {"document_id": self.document.id}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(document={self.document.name})"

    def _get_instance(self, **kwargs: Any) -> "TemporaryDocumentMention":
        return TemporaryDocumentMention(**kwargs)


class DocumentMention(Context, TemporaryDocumentMention):
    """A document ``Mention``."""

    __tablename__ = "document_mention"

    #: The unique id of the ``DocumentMention``.
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)

    #: The id of the parent ``Document``.
    document_id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"))

    #: The parent ``Document``.
    document = relationship("Context", foreign_keys=document_id)

    __table_args__ = (UniqueConstraint(document_id),)

    __mapper_args__ = {
        "polymorphic_identity": "document_mention",
        "inherit_condition": (id == Context.id),
    }

    def __init__(self, tc: TemporaryDocumentMention):
        self.stable_id = tc.get_stable_id()
        self.document = tc.document
