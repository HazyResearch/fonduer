"""Fonduer label model."""
from sqlalchemy import Column, Integer, String
from sqlalchemy.dialects import postgresql

from fonduer.meta import Meta
from fonduer.utils.models.annotation import AnnotationKeyMixin, AnnotationMixin


class GoldLabelKey(AnnotationKeyMixin, Meta.Base):
    """Gold label key class.

    A gold label's key that identifies the annotator of the gold label.
    """

    pass


class GoldLabel(AnnotationMixin, Meta.Base):
    """Gold label class.

    A separate class for labels from human annotators or other gold standards.
    """

    #: A list of integer values for each Key.
    values = Column(postgresql.ARRAY(Integer), nullable=False)


class LabelKey(AnnotationKeyMixin, Meta.Base):
    """Label key class.

    A label's key that identifies the labeling function.
    """

    pass


class Label(AnnotationMixin, Meta.Base):
    """Label class.

    A discrete label associated with a Candidate, indicating a target prediction value.

    Labels are used to represent the output of labeling functions. A Label's
    annotation key identifies the labeling function that provided the Label.
    """

    #: A list of integer values for each Key.
    values = Column(postgresql.ARRAY(Integer), nullable=False)


class StableLabel(Meta.Base):
    """Stable label table.

    A special secondary table for preserving labels created by *human
    annotators* in a stable format that does not cascade, and is independent of
    the Candidate IDs.

    .. note:: This is currently unused.
    """

    __tablename__ = "stable_label"

    #: Delimited list of the context stable ids.
    context_stable_ids = Column(
        String, primary_key=True
    )  # ~~ delimited list of the context stable ids

    #: The annotator's name
    annotator_name = Column(String, primary_key=True)

    #: Which split the label belongs to
    split = Column(Integer, default=0)

    # The value of the label
    value = Column(Integer, nullable=False)

    def __repr__(self) -> str:
        """Represent the stable label as a string."""
        return f"{self.__class__.__name__} ({self.annotator_name} : {self.value})"
