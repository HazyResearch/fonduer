from sqlalchemy import Column, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import relationship

from fonduer.candidates.models.temporary_context import TemporaryContext
from fonduer.parser.models.context import Context


class TemporaryFigureMention(TemporaryContext):
    """The TemporaryContext version of FigureMention."""

    def __init__(self, figure):
        super(TemporaryFigureMention, self).__init__()
        self.figure = figure  # The figure Context

    def __len__(self):
        return 1

    def __eq__(self, other):
        try:
            return self.figure == other.figure
        except AttributeError:
            return False

    def __ne__(self, other):
        try:
            return self.figure != other.figure
        except AttributeError:
            return True

    def __contains__(self, other_span):
        return self.__eq__(other_span)

    def __hash__(self):
        return hash(self.figure)

    def get_stable_id(self):
        """Return a stable id for the ``FigureMention``."""
        return "%s::%s:%s" % (
            self.figure.document.name,
            self._get_polymorphic_identity(),
            self.figure.position,
        )

    def _get_table(self):
        return FigureMention

    def _get_polymorphic_identity(self):
        return "figure_mention"

    def _get_insert_args(self):
        return {"figure_id": self.figure.id}

    def __repr__(self):
        return "{}(document={}, position={}, url={})".format(
            self.__class__.__name__,
            self.figure.document.name,
            self.figure.position,
            self.figure.url,
        )

    def _get_instance(self, **kwargs):
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

    def __gt__(self, other):
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()
