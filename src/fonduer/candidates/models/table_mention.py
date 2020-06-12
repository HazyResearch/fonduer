"""Fonduer table mention model."""
from typing import Any, Dict, Type

from sqlalchemy import Column, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import relationship

from fonduer.candidates.models.temporary_context import TemporaryContext
from fonduer.parser.models import Table
from fonduer.parser.models.context import Context
from fonduer.parser.models.utils import construct_stable_id


class TemporaryTableMention(TemporaryContext):
    """The TemporaryContext version of TableMention."""

    def __init__(self, table: Table) -> None:
        """Initialize TemporaryTableMention."""
        super().__init__()
        self.table = table  # The table Context

    def __len__(self) -> int:
        """Get the length of the mention."""
        return 1

    def __eq__(self, other: object) -> bool:
        """Check if the mention is equal to another mention."""
        if not isinstance(other, TemporaryTableMention):
            return NotImplemented
        return self.table == other.table

    def __ne__(self, other: object) -> bool:
        """Check if the mention is not equal to another mention."""
        if not isinstance(other, TemporaryTableMention):
            return NotImplemented
        return self.table != other.table

    def __gt__(self, other: object) -> bool:
        """Check if the mention is greater than another mention."""
        if not isinstance(other, TemporaryTableMention):
            return NotImplemented
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()

    def __contains__(self, other: object) -> bool:
        """Check if the mention contains another mention."""
        if not isinstance(other, TemporaryTableMention):
            return NotImplemented
        return self.__eq__(other)

    def __hash__(self) -> int:
        """Get the hash value of mention."""
        return hash(self.table)

    def get_stable_id(self) -> str:
        """Return a stable id."""
        return construct_stable_id(self.table, self._get_polymorphic_identity(), 0, 0)

    def _get_table(self) -> Type["TableMention"]:
        return TableMention

    def _get_polymorphic_identity(self) -> str:
        return "table_mention"

    def _get_insert_args(self) -> Dict[str, Any]:
        return {"table_id": self.table.id}

    def __repr__(self) -> str:
        """Represent the mention as a string."""
        return (
            f"{self.__class__.__name__}"
            f"("
            f"document={self.table.document.name}, "
            f"position={self.table.position}"
            f")"
        )

    def _get_instance(self, **kwargs: Any) -> "TemporaryTableMention":
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

    def __init__(self, tc: TemporaryTableMention):
        """Initialize TableMention."""
        self.stable_id = tc.get_stable_id()
        self.table = tc.table
