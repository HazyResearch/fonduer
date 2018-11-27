from sqlalchemy import Column, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import relationship

from fonduer.candidates.models.temporary_context import TemporaryContext
from fonduer.parser.models.context import Context


class TemporaryCellMention(TemporaryContext):
    """The TemporaryContext version of CellMention."""

    def __init__(self, cell):
        super(TemporaryCellMention, self).__init__()
        self.cell = cell  # The cell Context

    def __len__(self):
        return 1

    def __eq__(self, other):
        try:
            return self.cell == other.cell
        except AttributeError:
            return False

    def __ne__(self, other):
        try:
            return self.cell != other.cell
        except AttributeError:
            return True

    def __contains__(self, other_cell):
        return self.__eq__(other_cell)

    def __hash__(self):
        return hash(self.cell)

    def get_stable_id(self):
        """Return a stable id for the ``CellMention``."""
        return "%s::%s:%s:%s" % (
            self.cell.document.name,
            self._get_polymorphic_identity(),
            self.cell.table.position,
            self.cell.position,
        )

    def _get_table(self):
        return CellMention

    def _get_polymorphic_identity(self):
        return "cell_mention"

    def _get_insert_args(self):
        return {"cell_id": self.cell.id}

    def __repr__(self):
        return "{}(document={}, table_position={}, position={})".format(
            self.__class__.__name__,
            self.cell.document.name,
            self.cell.table.position,
            self.cell.position,
        )

    def _get_instance(self, **kwargs):
        return TemporaryCellMention(**kwargs)


class CellMention(Context, TemporaryCellMention):
    """A cell ``Mention``."""

    __tablename__ = "cell_mention"

    #: The unique id of the ``CellMention``.
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)

    #: The id of the parent ``Cell``.
    cell_id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"))

    #: The parent ``Cell``.
    cell = relationship("Context", foreign_keys=cell_id)

    __table_args__ = (UniqueConstraint(cell_id),)

    __mapper_args__ = {
        "polymorphic_identity": "cell_mention",
        "inherit_condition": (id == Context.id),
    }

    def __gt__(self, other):
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()
