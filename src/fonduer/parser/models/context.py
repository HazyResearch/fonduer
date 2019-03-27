from sqlalchemy import Column, Integer, String

from fonduer.meta import Meta

# Grab pointer to global metadata
_meta = Meta.init()


class Context(_meta.Base):
    """A piece of content from which Candidates are composed.

    This serves as the base class of the Fonduer document model.
    """

    __tablename__ = "context"

    #: The unique id of the ``Context``.
    id = Column(Integer, primary_key=True)

    #: The type of the ``Context`` represented as a string (e.g. "sentence",
    #: "paragraph", "figure").
    type = Column(String, nullable=False)

    #: A stable representation of the ``Context`` that will not change between
    #: runs.
    stable_id = Column(String, unique=True, nullable=False)

    __mapper_args__ = {"polymorphic_identity": "context", "polymorphic_on": type}
