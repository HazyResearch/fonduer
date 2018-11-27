from sqlalchemy import Column, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import relationship

from fonduer.candidates.models.temporary_context import TemporaryContext
from fonduer.parser.models.context import Context


class TemporaryParagraphMention(TemporaryContext):
    """The TemporaryContext version of ParagraphMention."""

    def __init__(self, paragraph):
        super(TemporaryParagraphMention, self).__init__()
        self.paragraph = paragraph  # The paragraph Context

    def __len__(self):
        return 1

    def __eq__(self, other):
        try:
            return self.paragraph == other.paragraph
        except AttributeError:
            return False

    def __ne__(self, other):
        try:
            return self.paragraph != other.paragraph
        except AttributeError:
            return True

    def __contains__(self, other_paragraph):
        return self.__eq__(other_paragraph)

    def __hash__(self):
        return hash(self.paragraph)

    def get_stable_id(self):
        """Return a stable id for the ``ParagraphMention``."""
        return "%s::%s:%s" % (
            self.paragraph.document.name,
            self._get_polymorphic_identity(),
            self.paragraph.position,
        )

    def _get_table(self):
        return ParagraphMention

    def _get_polymorphic_identity(self):
        return "paragraph_mention"

    def _get_insert_args(self):
        return {"paragraph_id": self.paragraph.id}

    def __repr__(self):
        return "{}(document={}, position={})".format(
            self.__class__.__name__,
            self.paragraph.document.name,
            self.paragraph.position,
        )

    def _get_instance(self, **kwargs):
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

    def __gt__(self, other):
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()
