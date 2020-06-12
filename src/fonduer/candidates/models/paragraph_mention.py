"""Fonduer paragraph mention model."""
from typing import Any, Dict, Type

from sqlalchemy import Column, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import relationship

from fonduer.candidates.models.temporary_context import TemporaryContext
from fonduer.parser.models import Paragraph
from fonduer.parser.models.context import Context
from fonduer.parser.models.utils import construct_stable_id


class TemporaryParagraphMention(TemporaryContext):
    """The TemporaryContext version of ParagraphMention."""

    def __init__(self, paragraph: Paragraph) -> None:
        """Initialize TemporaryParagraphMention."""
        super().__init__()
        self.paragraph = paragraph  # The paragraph Context

    def __len__(self) -> int:
        """Get the length of the mention."""
        return 1

    def __eq__(self, other: object) -> bool:
        """Check if the mention is equal to another mention."""
        if not isinstance(other, TemporaryParagraphMention):
            return NotImplemented
        return self.paragraph == other.paragraph

    def __ne__(self, other: object) -> bool:
        """Check if the mention is not equal to another mention."""
        if not isinstance(other, TemporaryParagraphMention):
            return NotImplemented
        return self.paragraph != other.paragraph

    def __gt__(self, other: object) -> bool:
        """Check if the mention is greater than another mention."""
        if not isinstance(other, TemporaryParagraphMention):
            return NotImplemented
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()

    def __contains__(self, other: object) -> bool:
        """Check if the mention contains another mention."""
        if not isinstance(other, TemporaryParagraphMention):
            return NotImplemented
        return self.__eq__(other)

    def __hash__(self) -> int:
        """Get the hash value of mention."""
        return hash(self.paragraph)

    def get_stable_id(self) -> str:
        """Return a stable id."""
        return construct_stable_id(
            self.paragraph, self._get_polymorphic_identity(), 0, 0
        )

    def _get_table(self) -> Type["ParagraphMention"]:
        return ParagraphMention

    def _get_polymorphic_identity(self) -> str:
        return "paragraph_mention"

    def _get_insert_args(self) -> Dict[str, Any]:
        return {"paragraph_id": self.paragraph.id}

    def __repr__(self) -> str:
        """Represent the mention as a string."""
        return (
            f"{self.__class__.__name__}"
            f"("
            f"document={self.paragraph.document.name}, "
            f"position={self.paragraph.position}"
            f")"
        )

    def _get_instance(self, **kwargs: Any) -> "TemporaryParagraphMention":
        return TemporaryParagraphMention(**kwargs)


class ParagraphMention(Context, TemporaryParagraphMention):
    """A paragraph ``Mention``."""

    __tablename__ = "paragraph_mention"

    #: The unique id of the ``ParagraphMention``.
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)

    #: The id of the parent ``Paragraph``.
    paragraph_id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"))

    #: The parent ``Paragraph``.
    paragraph = relationship("Context", foreign_keys=paragraph_id)

    __table_args__ = (UniqueConstraint(paragraph_id),)

    __mapper_args__ = {
        "polymorphic_identity": "paragraph_mention",
        "inherit_condition": (id == Context.id),
    }

    def __init__(self, tc: TemporaryParagraphMention):
        """Initialize ParagraphMention."""
        self.stable_id = tc.get_stable_id()
        self.paragraph = tc.paragraph
