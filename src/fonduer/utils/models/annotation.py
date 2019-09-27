from typing import Tuple

from sqlalchemy import Column, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import backref, relationship

from fonduer.utils.utils import camel_to_under


class AnnotationKeyMixin(object):
    """Mixin class for defining annotation key tables.

    An AnnotationKey is the unique name associated with a set of Annotations,
    corresponding e.g. to a single labeling function or feature.
    """

    __name__: str  # declare for mypy

    @declared_attr
    def __tablename__(cls) -> str:
        return camel_to_under(cls.__name__)

    @declared_attr
    def name(cls) -> Column:
        """The name of the Key."""
        return Column(String, primary_key=True)

    @declared_attr
    def candidate_classes(cls) -> Column:
        """The name of the Key."""
        return Column(postgresql.ARRAY(String), nullable=False)

    @declared_attr
    def __table_args__(cls) -> Tuple[UniqueConstraint]:
        return (UniqueConstraint("name"),)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.name})"


class AnnotationMixin(object):
    """Mixin class for defining annotation tables.

    An annotation is a value associated with a Candidate. Examples include
    labels, features, and predictions. New types of annotations can be defined
    by creating an annotation class and corresponding annotation, for example:

    .. code-block:: python

        from fonduer.utils.models import AnnotationMixin
        from fonduer.meta import Meta

        class NewAnnotation(AnnotationMixin, Meta.Base):
            values = Column(Float, nullable=False)

    The annotation class should include a Column attribute named values.
    """

    __name__: str  # declare for mypy
    values: Column  # declare for mypy

    @declared_attr
    def __tablename__(cls) -> str:
        return camel_to_under(cls.__name__)

    # The key is the "name" or "type" of the Annotation- e.g. the name of a
    # feature, lf, or of a human annotator
    @declared_attr
    def keys(cls) -> Column:
        """A list of strings of each Key name."""
        return Column(postgresql.ARRAY(String), nullable=False)

    # Every annotation is with respect to a candidate
    @declared_attr
    def candidate_id(cls) -> Column:
        """The id of the ``Candidate`` being annotated."""
        return Column(
            "candidate_id",
            Integer,
            ForeignKey("candidate.id", ondelete="CASCADE"),
            primary_key=True,
        )

    @declared_attr
    def candidate(cls) -> relationship:
        """The ``Candidate``."""
        return relationship(
            "Candidate",
            backref=backref(
                camel_to_under(cls.__name__) + "s",
                cascade="all, delete-orphan",
                cascade_backrefs=False,
            ),
            cascade_backrefs=False,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f" ("
            f"{self.keys}"
            f" = "
            f"{self.values}"
            f")"
        )
