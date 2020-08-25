"""Fonduer candidate model."""
import logging
from typing import Any, Dict, List, Optional, Tuple, Type

from sqlalchemy import Column, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import backref, relationship

from fonduer.candidates.models.mention import Mention
from fonduer.meta import Meta
from fonduer.utils.utils import camel_to_under

logger = logging.getLogger(__name__)

# This global dictionary contains all classes that have been declared in this
# Python environment, so that candidate_subclass() can return a class if it
# already exists and is identical in specification to the requested class
candidate_subclasses: Dict[str, Tuple] = {}


class Candidate(Meta.Base):
    """
    An abstract candidate relation.

    New relation types should be defined by calling candidate_subclass(),
    **not** subclassing this class directly.
    """

    __tablename__ = "candidate"

    #: The unique id for the ``Candidate``.
    id = Column(Integer, primary_key=True)

    #: The type for the ``Candidate``, which corresponds to the names the user
    #: gives to the candidate_subclasses.
    type = Column(String, nullable=False)

    #: Which split the ``Candidate`` belongs to. Used to organize train/dev/test.
    split = Column(Integer, nullable=False, default=0, index=True)

    __mapper_args__ = {"polymorphic_identity": "candidate", "polymorphic_on": type}

    # __table_args__ = {"extend_existing" : True}

    def get_mentions(self) -> Tuple[Mention, ...]:
        """Get a tuple of the constituent ``Mentions`` making up this ``Candidate``."""
        return tuple(getattr(self, name) for name in self.__argnames__)

    def __len__(self) -> int:
        """Get the length of the candidate."""
        return len(self.__argnames__)

    def __getitem__(self, key: int) -> Mention:
        """Get the mention from candidate."""
        return self.get_mentions()[key]

    def __repr__(self) -> str:
        """Represent the candidate as a string."""
        return (
            f"{self.__class__.__name__}"
            f"("
            f"{', '.join(map(str, self.get_mentions()))}"
            f")"
        )

    def __gt__(self, other_cand: "Candidate") -> bool:
        """Check if the candidate is greater than another candidate."""
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other_cand.__repr__()


def candidate_subclass(
    class_name: str,
    args: List[Mention],
    table_name: Optional[str] = None,
    cardinality: Optional[int] = None,
    values: Optional[List[Any]] = None,
    nullables: Optional[List[bool]] = None,
) -> Type[Candidate]:
    """Create new relation.

    Creates and returns a Candidate subclass with provided argument names,
    which are Context type. Creates the table in DB if does not exist yet.

    Import using:

    .. code-block:: python

        from fonduer.candidates.models import candidate_subclass

    :param class_name: The name of the class, should be "camel case" e.g.
        NewCandidate
    :param args: A list of names of constituent arguments, which refer to the
        Contexts--representing mentions--that comprise the candidate
    :param table_name: The name of the corresponding table in DB; if not
        provided, is converted from camel case by default, e.g. new_candidate
    :param cardinality: The cardinality of the variable corresponding to the
        Candidate. By default is 2 i.e. is a binary value, e.g. is or is not
        a true mention.
    :param values: A list of values a candidate can take as their label.
    :param nullables: The number of nullables must match that of args.
        If nullables[i]==True, a mention for ith mention subclass can be NULL.
        If nullables=``None`` (by default), no mention can be NULL.
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

    if nullables:
        if len(nullables) != len(args):
            raise ValueError("The number of nullables must match that of args.")
    else:
        nullables = [False] * len(args)

    class_spec = (args, table_name, cardinality, values)
    if class_name in candidate_subclasses:
        if class_spec == candidate_subclasses[class_name][1]:
            return candidate_subclasses[class_name][0]
        else:
            raise ValueError(
                f"Candidate subclass {class_name} "
                f"already exists in memory with incompatible "
                f"specification: {candidate_subclasses[class_name][1]}"
            )
    else:
        # Set the class attributes == the columns in the database
        class_attribs = {
            # Declares name for storage table
            "__tablename__": table_name,
            # Connects candidate_subclass records to generic Candidate records
            "id": Column(
                Integer,
                ForeignKey("candidate.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            # Store values & cardinality information in the class only
            "values": values,
            "cardinality": cardinality,
            # Polymorphism information for SQLAlchemy
            "__mapper_args__": {"polymorphic_identity": table_name},
            # Helper method to get argument names
            "__argnames__": [_.__tablename__ for _ in args],
            "mentions": args,
            "nullables": nullables,
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
        for arg, nullable in zip(args, nullables):
            # Primary arguments are constituent Contexts, and their ids
            class_attribs[arg.__tablename__ + "_id"] = Column(
                Integer,
                ForeignKey(arg.__tablename__ + ".id", ondelete="CASCADE"),
                nullable=nullable,
            )
            class_attribs[arg.__tablename__] = relationship(
                arg.__name__,
                backref=backref(
                    table_name + "_" + arg.__tablename__ + "s",
                    cascade_backrefs=False,
                    cascade="all, delete-orphan",
                ),
                cascade_backrefs=False,
                foreign_keys=class_attribs[arg.__tablename__ + "_id"],
            )
            unique_args.append(class_attribs[arg.__tablename__ + "_id"])

        # Add unique constraints to the arguments
        class_attribs["__table_args__"] = (UniqueConstraint(*unique_args),)

        # Create class
        C = type(class_name, (Candidate,), class_attribs)

        # Create table in DB
        if Meta.engine and not Meta.engine.has_table(table_name):
            C.__table__.create(bind=Meta.engine)  # type: ignore

        candidate_subclasses[class_name] = C, class_spec
        # Make this dynamically created class picklable
        # https://stackoverflow.com/a/39529149
        globals()[class_name] = C

        return C
