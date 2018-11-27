from sqlalchemy import Column, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import relationship

from fonduer.candidates.models.temporary_context import TemporaryContext
from fonduer.parser.models.context import Context


class TemporarySectionMention(TemporaryContext):
    """The TemporaryContext version of SectionMention."""

    def __init__(self, section):
        super(TemporarySectionMention, self).__init__()
        self.section = section  # The section Context

    def __len__(self):
        return 1

    def __eq__(self, other):
        try:
            return self.section == other.section
        except AttributeError:
            return False

    def __ne__(self, other):
        try:
            return self.section != other.section
        except AttributeError:
            return True

    def __contains__(self, other_section):
        return self.__eq__(other_section)

    def __hash__(self):
        return hash(self.section)

    def get_stable_id(self):
        """Return a stable id for the ``SectionMention``."""
        return "%s::%s:%s" % (
            self.section.document.name,
            self._get_polymorphic_identity(),
            self.section.position,
        )

    def _get_table(self):
        return SectionMention

    def _get_polymorphic_identity(self):
        return "section_mention"

    def _get_insert_args(self):
        return {"section_id": self.section.id}

    def __repr__(self):
        return "{}(document={}, position={})".format(
            self.__class__.__name__, self.section.document.name, self.section.position
        )

    def _get_instance(self, **kwargs):
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

    def __gt__(self, other):
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()
