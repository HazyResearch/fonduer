"""Fonduer section mention model."""
from typing import Any, Dict, Type

from sqlalchemy import Column, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import relationship

from fonduer.candidates.models.temporary_context import TemporaryContext
from fonduer.parser.models import Section
from fonduer.parser.models.context import Context
from fonduer.parser.models.utils import construct_stable_id


class TemporarySectionMention(TemporaryContext):
    """The TemporaryContext version of SectionMention."""

    def __init__(self, section: Section) -> None:
        """Initialize TemporarySectionMention."""
        super().__init__()
        self.section = section  # The section Context

    def __len__(self) -> int:
        """Get the length of the mention."""
        return 1

    def __eq__(self, other: object) -> bool:
        """Check if the mention is equal to another mention."""
        if not isinstance(other, TemporarySectionMention):
            return NotImplemented
        return self.section == other.section

    def __ne__(self, other: object) -> bool:
        """Check if the mention is not equal to another mention."""
        if not isinstance(other, TemporarySectionMention):
            return NotImplemented
        return self.section != other.section

    def __gt__(self, other: object) -> bool:
        """Check if the mention is greater than another mention."""
        if not isinstance(other, TemporarySectionMention):
            return NotImplemented
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()

    def __contains__(self, other: object) -> bool:
        """Check if the mention contains another mention."""
        if not isinstance(other, TemporarySectionMention):
            return NotImplemented
        return self.__eq__(other)

    def __hash__(self) -> int:
        """Get the hash value of mention."""
        return hash(self.section)

    def get_stable_id(self) -> str:
        """Return a stable id."""
        return construct_stable_id(self.section, self._get_polymorphic_identity(), 0, 0)

    def _get_table(self) -> Type["SectionMention"]:
        return SectionMention

    def _get_polymorphic_identity(self) -> str:
        return "section_mention"

    def _get_insert_args(self) -> Dict[str, Any]:
        return {"section_id": self.section.id}

    def __repr__(self) -> str:
        """Represent the mention as a string."""
        return (
            f"{self.__class__.__name__}"
            f"("
            f"document={self.section.document.name}, "
            f"position={self.section.position}"
            f")"
        )

    def _get_instance(self, **kwargs: Any) -> "TemporarySectionMention":
        return TemporarySectionMention(**kwargs)


class SectionMention(Context, TemporarySectionMention):
    """A section ``Mention``."""

    __tablename__ = "section_mention"

    #: The unique id of the ``SectionMention``.
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)

    #: The id of the parent ``Section``.
    section_id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"))

    #: The parent ``Section``.
    section = relationship("Context", foreign_keys=section_id)

    __table_args__ = (UniqueConstraint(section_id),)

    __mapper_args__ = {
        "polymorphic_identity": "section_mention",
        "inherit_condition": (id == Context.id),
    }

    def __init__(self, tc: TemporarySectionMention):
        """Initialize SectionMention."""
        self.stable_id = tc.get_stable_id()
        self.section = tc.section
