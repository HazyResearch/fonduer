"""Fonduer figure mention model."""
from typing import Any, Dict, Type

from sqlalchemy import Column, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import relationship

from fonduer.candidates.models.temporary_context import TemporaryContext
from fonduer.parser.models import Figure
from fonduer.parser.models.context import Context
from fonduer.parser.models.utils import construct_stable_id


class TemporaryFigureMention(TemporaryContext):
    """The TemporaryContext version of FigureMention."""

    def __init__(self, figure: Figure) -> None:
        """Initialize TemporaryFigureMention."""
        super().__init__()
        self.figure = figure  # The figure Context

    def __len__(self) -> int:
        """Get the length of the mention."""
        return 1

    def __eq__(self, other: object) -> bool:
        """Check if the mention is equal to another mention."""
        if not isinstance(other, TemporaryFigureMention):
            return NotImplemented
        return self.figure == other.figure

    def __ne__(self, other: object) -> bool:
        """Check if the mention is not equal to another mention."""
        if not isinstance(other, TemporaryFigureMention):
            return NotImplemented
        return self.figure != other.figure

    def __gt__(self, other: object) -> bool:
        """Check if the mention is greater than another mention."""
        if not isinstance(other, TemporaryFigureMention):
            return NotImplemented
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()

    def __contains__(self, other: object) -> bool:
        """Check if the mention contains another mention."""
        if not isinstance(other, TemporaryFigureMention):
            return NotImplemented
        return self.__eq__(other)

    def __hash__(self) -> int:
        """Get the hash value of mention."""
        return hash(self.figure)

    def get_stable_id(self) -> str:
        """Return a stable id."""
        return construct_stable_id(self.figure, self._get_polymorphic_identity(), 0, 0)

    def _get_table(self) -> Type["FigureMention"]:
        return FigureMention

    def _get_polymorphic_identity(self) -> str:
        return "figure_mention"

    def _get_insert_args(self) -> Dict[str, Any]:
        return {"figure_id": self.figure.id}

    def __repr__(self) -> str:
        """Represent the mention as a string."""
        return (
            f"{self.__class__.__name__}"
            f"("
            f"document={self.figure.document.name}, "
            f"position={self.figure.position}, "
            f"url={self.figure.url}"
            f")"
        )

    def _get_instance(self, **kwargs: Any) -> "TemporaryFigureMention":
        return TemporaryFigureMention(**kwargs)


class FigureMention(Context, TemporaryFigureMention):
    """A figure ``Mention``."""

    __tablename__ = "figure_mention"

    #: The unique id of the ``FigureMention``.
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)

    #: The id of the parent ``Figure``.
    figure_id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"))

    #: The parent ``Figure``.
    figure = relationship("Context", foreign_keys=figure_id)

    __table_args__ = (UniqueConstraint(figure_id),)

    __mapper_args__ = {
        "polymorphic_identity": "figure_mention",
        "inherit_condition": (id == Context.id),
    }

    def __init__(self, tc: TemporaryFigureMention):
        """Initialize FigureMention."""
        self.stable_id = tc.get_stable_id()
        self.figure = tc.figure
