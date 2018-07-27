import logging

from sqlalchemy import Column, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import backref, relationship

from fonduer.meta import Meta
from fonduer.utils import camel_to_under

_meta = Meta.init()
logger = logging.getLogger(__name__)


class Mention(_meta.Base):
    """
    An abstract Mention.

    New mention types should be defined by calling mention_subclass(),
    **not** subclassing this class directly.
    """

    __tablename__ = "mention"
    id = Column(Integer, primary_key=True)
    type = Column(String, nullable=False)
    split = Column(Integer, nullable=False, default=0, index=True)

    __mapper_args__ = {"polymorphic_identity": "mention", "polymorphic_on": type}

    def get_contexts(self):
        """Get the consituent context making up this mention."""
        return tuple(getattr(self, name) for name in self.__argnames__)

    def get_parent(self):
        # Fails if both contexts don't have same parent
        p = [c.get_parent() for c in self.get_contexts()]
        if p.count(p[0]) == len(p):
            return p[0]
        else:
            raise Exception("Contexts do not all have same parent")

    def get_cids(self):
        """Get the canonical IDs (CIDs) of the context making up this mention."""
        return tuple(getattr(self, name + "_cid") for name in self.__argnames__)

    def __len__(self):
        return len(self.__argnames__)

    def __getitem__(self, key):
        return self.get_contexts()[key]

    def __repr__(self):
        return "%s(%s)" % (
            self.__class__.__name__,
            ", ".join(map(str, self.get_contexts())),
        )

    def __gt__(self, other_cand):
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other_cand.__repr__()


# This global dictionary contains all classes that have been declared in this
# Python environment, so that mention_subclass() can return a class if it
# already exists and is identical in specification to the requested class
mention_subclasses = {}


def mention_subclass(class_name, args, table_name=None):
    """
    Creates and returns a Mention subclass with provided argument names,
    which are Context type. Creates the table in DB if does not exist yet.

    Import using:

    .. code-block:: python

        from fonduer.candidates.models import mention_subclass

    :param class_name: The name of the class, should be "camel case" e.g.
        NewMention
    :param args: A list of names of consituent arguments, which refer to the
        Contexts--representing mentions--that comprise the mention
    :param table_name: The name of the corresponding table in DB; if not
        provided, is converted from camel case by default, e.g. new_mention
    """
    if table_name is None:
        table_name = camel_to_under(class_name)

    class_spec = (args, table_name)
    if class_name in mention_subclasses:
        if class_spec == mention_subclasses[class_name][1]:
            return mention_subclasses[class_name][0]
        else:
            raise ValueError(
                "Mention subclass "
                + class_name
                + " already exists in memory with incompatible "
                + "specification: "
                + str(mention_subclasses[class_name][1])
            )
    else:
        # Set the class attributes == the columns in the database
        class_attribs = {
            # Declares name for storage table
            "__tablename__": table_name,
            # Connects mention_subclass records to generic Mention records
            "id": Column(
                Integer, ForeignKey("mention.id", ondelete="CASCADE"), primary_key=True
            ),
            # Polymorphism information for SQLAlchemy
            "__mapper_args__": {"polymorphic_identity": table_name},
            # Helper method to get argument names
            "__argnames__": args,
        }

        # Create named arguments, i.e. the entity mentions comprising the
        # relation mention. For each entity mention: id, cid ("canonical id"),
        # and pointer to Context
        unique_args = []
        for arg in args:

            # Primary arguments are constituent Contexts, and their ids
            class_attribs[arg + "_id"] = Column(
                Integer, ForeignKey("context.id", ondelete="CASCADE")
            )
            class_attribs[arg] = relationship(
                "Context",
                backref=backref(
                    table_name + "_" + arg + "s",
                    cascade_backrefs=False,
                    cascade="all, delete-orphan",
                ),
                cascade_backrefs=False,
                foreign_keys=class_attribs[arg + "_id"],
            )
            unique_args.append(class_attribs[arg + "_id"])

            # Canonical ids, to be set post-entity normalization stage
            class_attribs[arg + "_cid"] = Column(String)

        # Add unique constraints to the arguments
        class_attribs["__table_args__"] = (UniqueConstraint(*unique_args),)

        # Create class
        C = type(class_name, (Mention,), class_attribs)

        # Create table in DB
        if not _meta.engine.dialect.has_table(_meta.engine, table_name):
            C.__table__.create(bind=_meta.engine)

        mention_subclasses[class_name] = C, class_spec

        return C
