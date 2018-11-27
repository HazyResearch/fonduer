from sqlalchemy import Column, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import relationship

from fonduer.candidates.models.temporary_context import TemporaryContext
from fonduer.parser.models.context import Context


class TemporaryCaptionMention(TemporaryContext):
    """The TemporaryContext version of CaptionMention."""

    def __init__(self, caption):
        super(TemporaryCaptionMention, self).__init__()
        self.caption = caption  # The caption Context

    def __len__(self):
        return 1

    def __eq__(self, other):
        try:
            return self.caption == other.caption
        except AttributeError:
            return False

    def __ne__(self, other):
        try:
            return self.caption != other.caption
        except AttributeError:
            return True

    def __contains__(self, other_caption):
        return self.__eq__(other_caption)

    def __hash__(self):
        return hash(self.caption)

    def get_stable_id(self):
        """Return a stable id for the ``CaptionMention``."""
        return "%s::%s:%s" % (
            self.caption.document.name,
            self._get_polymorphic_identity(),
            self.caption.position,
        )

    def _get_table(self):
        return CaptionMention

    def _get_polymorphic_identity(self):
        return "caption_mention"

    def _get_insert_args(self):
        return {"caption_id": self.caption.id}

    def __repr__(self):
        return "{}(document={}, position={})".format(
            self.__class__.__name__, self.caption.document.name, self.caption.position
        )

    def _get_instance(self, **kwargs):
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

    def __gt__(self, other):
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()
