from typing import Any, Dict, Type

from sqlalchemy import Column, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import relationship

from fonduer.candidates.models.temporary_context import TemporaryContext
from fonduer.parser.models import Section
from fonduer.parser.models.context import Context


class TemporarySectionMention(TemporaryContext):
    """The TemporaryContext version of SectionMention."""

    def __init__(self, section: Section) -> None:
        super().__init__()
        self.section = section  # The section Context

    def __len__(self) -> int:
        return 1

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TemporarySectionMention):
            return NotImplemented
        return self.section == other.section

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, TemporarySectionMention):
            return NotImplemented
        return self.section != other.section

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, TemporarySectionMention):
            return NotImplemented
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()

    def __contains__(self, other: object) -> bool:
        if not isinstance(other, TemporarySectionMention):
            return NotImplemented
        return self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self.section)

    def get_stable_id(self) -> str:
        """Return a stable id for the ``SectionMention``."""
        return self.section.stable_id

    def _get_table(self) -> Type["SectionMention"]:
        return SectionMention

    def _get_polymorphic_identity(self) -> str:
        return "section_mention"

    def _get_insert_args(self) -> Dict[str, Any]:
        return {"section_id": self.section.id}

    def __repr__(self) -> str:
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
