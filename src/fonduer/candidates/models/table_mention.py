from sqlalchemy import Column, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import relationship

from fonduer.candidates.models.temporary_context import TemporaryContext
from fonduer.parser.models.context import Context


class TemporaryTableMention(TemporaryContext):
    """The TemporaryContext version of TableMention."""

    def __init__(self, table):
        super(TemporaryTableMention, self).__init__()
        self.table = table  # The table Context

    def __len__(self):
        return 1

    def __eq__(self, other):
        try:
            return self.table == other.table
        except AttributeError:
            return False

    def __ne__(self, other):
        try:
            return self.table != other.table
        except AttributeError:
            return True

    def __contains__(self, other_table):
        return self.__eq__(other_table)

    def __hash__(self):
        return hash(self.table)

    def get_stable_id(self):
        """Return a stable id for the ``TableMention``."""
        return "%s::%s:%s" % (
            self.table.document.name,
            self._get_polymorphic_identity(),
            self.table.position,
        )

    def _get_table(self):
        return TableMention

    def _get_polymorphic_identity(self):
        return "table_mention"

    def _get_insert_args(self):
        return {"table_id": self.table.id}

    def __repr__(self):
        return "{}(document={}, position={})".format(
            self.__class__.__name__, self.table.document.name, self.table.position
        )

    def _get_instance(self, **kwargs):
        return TemporaryTableMention(**kwargs)


class TableMention(Context, TemporaryTableMention):
    """A table ``Mention``."""

    __tablename__ = "table_mention"

    #: The unique id of the ``TableMention``.
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)

    #: The id of the parent ``Table``.
    table_id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"))

    #: The parent ``Table``.
    table = relationship("Context", foreign_keys=table_id)

    __table_args__ = (UniqueConstraint(table_id),)

    __mapper_args__ = {
        "polymorphic_identity": "table_mention",
        "inherit_condition": (id == Context.id),
    }

    def __gt__(self, other):
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()
