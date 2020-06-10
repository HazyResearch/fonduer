"""Fonduer mention model."""
import logging
from typing import Any, Dict, List, Optional, Tuple, Type

from sqlalchemy import Column, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import backref, relationship

from fonduer.meta import Meta
from fonduer.parser.models import Context
from fonduer.utils.utils import camel_to_under

logger = logging.getLogger(__name__)

# This global dictionary contains all classes that have been declared in this
# Python environment, so that mention_subclass() can return a class if it
# already exists and is identical in specification to the requested class.
mention_subclasses: Dict[str, Tuple] = {}


class Mention(Meta.Base):
    """
    An abstract Mention.

    New mention types should be defined by calling mention_subclass(),
    **not** subclassing this class directly.
    """

    __tablename__ = "mention"

    #: The unique id of the ``Mention``.
    id = Column(Integer, primary_key=True)

    #: The type for the ``Mention``, which corresponds to the names the user
    #: gives to the mention_subclass.
    type = Column(String, nullable=False)

    __mapper_args__ = {"polymorphic_identity": "mention", "polymorphic_on": type}

    def get_contexts(self) -> Tuple[Context, ...]:
        """Get the constituent context making up this mention."""
        return tuple(getattr(self, name) for name in self.__argnames__)

    def __len__(self) -> int:
        """Get the length of the mention."""
        return len(self.__argnames__)

    def __getitem__(self, key: int) -> Context:
        """Get the context from mention."""
        return self.get_contexts()[key]

    def __repr__(self) -> str:
        """Represent the mention as a string."""
        return (
            f"{self.__class__.__name__}"
            f"("
            f"{', '.join(map(str, self.get_contexts()))}"
            f")"
        )

    def __gt__(self, other: "Mention") -> bool:
        """Check if the mention is greater than another mention."""
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()


def mention_subclass(
    class_name: str,
    cardinality: Optional[int] = None,
    values: Optional[List[Any]] = None,
    table_name: Optional[str] = None,
) -> Type[Mention]:
    """Create new mention.

    Creates and returns a Mention subclass with provided argument names,
    which are Context type. Creates the table in DB if does not exist yet.

    Import using:

    .. code-block:: python

        from fonduer.candidates.models import mention_subclass

    :param class_name: The name of the class, should be "camel case" e.g.
        NewMention
    :param table_name: The name of the corresponding table in DB; if not
        provided, is converted from camel case by default, e.g. new_mention
    :param values: The values that the variable corresponding to the Mention
        can take. By default it will be [True, False].
    :param cardinality: The cardinality of the variable corresponding to the
        Mention. By default is 2 i.e. is a binary value, e.g. is or is not
        a true mention.
    """
    if table_name is None:
        table_name = camel_to_under(class_name)

    # If cardinality and values are None, default to binary classification
    if cardinality is None and values is None:
        values = [True, False]
        cardinality = 2
    # Else use values if present, and validate proper input
    elif values is not None:
        if cardinality is not None and len(values) != cardinality:
            raise ValueError("Number of values must match cardinality.")
        if None in values:
            raise ValueError("`None` is a protected value.")
        # Note that bools are instances of ints in Python...
        if any([isinstance(v, int) and not isinstance(v, bool) for v in values]):
            raise ValueError(
                (
                    "Default usage of values is consecutive integers."
                    "Leave values unset if trying to define values as integers."
                )
            )
        cardinality = len(values)

    # If cardinality is specified but not values, fill in with ints
    elif cardinality is not None:
        values = list(range(cardinality))

    args = ["context"]
    class_spec = (args, table_name, cardinality, values)
    if class_name in mention_subclasses:
        if class_spec == mention_subclasses[class_name][1]:
            return mention_subclasses[class_name][0]
        else:
            raise ValueError(
                f"Mention subclass {class_name} "
                f"already exists in memory with incompatible "
                f"specification: {mention_subclasses[class_name][1]}"
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
            # Store values & cardinality information in the class only
            "values": values,
            "cardinality": cardinality,
            # Polymorphism information for SQLAlchemy
            "__mapper_args__": {"polymorphic_identity": table_name},
            # Helper method to get argument names
            "__argnames__": args,
        }
        class_attribs["document_id"] = Column(
            Integer, ForeignKey("document.id", ondelete="CASCADE")
        )
        class_attribs["document"] = relationship(
            "Document",
            backref=backref(table_name + "s", cascade="all, delete-orphan"),
            foreign_keys=class_attribs["document_id"],
        )

        # Create named arguments, i.e. the entity mentions comprising the
        # relation mention.
        unique_args = []
        for arg in args:

            # Primary arguments are constituent Contexts, and their ids
            class_attribs[arg + "_id"] = Column(
                Integer, ForeignKey("context.id", ondelete="CASCADE")
            )
            class_attribs[arg] = relationship(
                "Context", foreign_keys=class_attribs[arg + "_id"]
            )
            unique_args.append(class_attribs[arg + "_id"])

        # Add unique constraints to the arguments
        class_attribs["__table_args__"] = (UniqueConstraint(*unique_args),)

        # Create class
        C = type(class_name, (Mention,), class_attribs)

        # Create table in DB
        if Meta.engine and not Meta.engine.has_table(table_name):
            C.__table__.create(bind=Meta.engine)  # type: ignore

        mention_subclasses[class_name] = C, class_spec
        # Make this dynamically created class picklable
        # https://stackoverflow.com/a/39529149
        globals()[class_name] = C

        return C
