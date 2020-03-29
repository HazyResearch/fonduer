from typing import Any, Dict, Type

from sqlalchemy import Column, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import relationship

from fonduer.candidates.models.temporary_context import TemporaryContext
from fonduer.parser.models import Caption
from fonduer.parser.models.context import Context
from fonduer.parser.models.utils import construct_stable_id


class TemporaryCaptionMention(TemporaryContext):
    """The TemporaryContext version of CaptionMention."""

    def __init__(self, caption: Caption) -> None:
        super().__init__()
        self.caption = caption  # The caption Context

    def __len__(self) -> int:
        return 1

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TemporaryCaptionMention):
            return NotImplemented
        return self.caption == other.caption

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, TemporaryCaptionMention):
            return NotImplemented
        return self.caption != other.caption

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, TemporaryCaptionMention):
            return NotImplemented
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()

    def __contains__(self, other: object) -> bool:
        if not isinstance(other, TemporaryCaptionMention):
            return NotImplemented
        return self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self.caption)

    def get_stable_id(self) -> str:
        """
        Return a stable id.

        :rtype: string
        """
        return construct_stable_id(self.caption, self._get_polymorphic_identity(), 0, 0)

    def _get_table(self) -> Type["CaptionMention"]:
        return CaptionMention

    def _get_polymorphic_identity(self) -> str:
        return "caption_mention"

    def _get_insert_args(self) -> Dict[str, Any]:
        return {"caption_id": self.caption.id}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"document={self.caption.document.name}, "
            f"position={self.caption.position}"
            f")"
        )

    def _get_instance(self, **kwargs: Any) -> "TemporaryCaptionMention":
        return TemporaryCaptionMention(**kwargs)


class CaptionMention(Context, TemporaryCaptionMention):
    """A caption ``Mention``."""

    __tablename__ = "caption_mention"

    #: The unique id of the ``CaptionMention``.
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)

    #: The id of the parent ``Caption``.
    caption_id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"))

    #: The parent ``Caption``.
    caption = relationship("Context", foreign_keys=caption_id)

    __table_args__ = (UniqueConstraint(caption_id),)

    __mapper_args__ = {
        "polymorphic_identity": "caption_mention",
        "inherit_condition": (id == Context.id),
    }

    def __init__(self, tc: TemporaryCaptionMention):
        self.stable_id = tc.get_stable_id()
        self.caption = tc.caption
