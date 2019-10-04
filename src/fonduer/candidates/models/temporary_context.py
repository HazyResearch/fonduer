from builtins import object
from typing import Any, Dict, Type

from fonduer.parser.models.context import Context


class TemporaryContext(object):
    """
    A context which does not incur the overhead of a proper ORM-based Context
    object. The TemporaryContext class is specifically for the candidate
    extraction process, during which a MentionSpace object will generate many
    TemporaryContexts, which will then be filtered by Matchers prior to
    materialization of Mentions and constituent Context objects.

    Every Context object has a corresponding TemporaryContext object from which
    it inherits.

    A TemporaryContext must have specified equality / set membership semantics,
    a stable_id for checking uniqueness against the database, and a promote()
    method which returns a corresponding Context object.
    """

    def __init__(self) -> None:
        self.id = None

    def __repr__(self) -> str:
        raise NotImplementedError()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TemporaryContext):
            return NotImplemented
        raise NotImplementedError()

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, TemporaryContext):
            return NotImplemented
        raise NotImplementedError()

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, TemporaryContext):
            return NotImplemented
        raise NotImplementedError()

    def __contains__(self, other: object) -> bool:
        if not isinstance(other, TemporaryContext):
            return NotImplemented
        raise NotImplementedError()

    def __hash__(self) -> int:
        raise NotImplementedError()

    def _get_polymorphic_identity(self) -> str:
        raise NotImplementedError()

    def get_stable_id(self) -> str:
        raise NotImplementedError()

    def _get_table(self) -> Type[Context]:
        raise NotImplementedError()

    def _get_insert_args(self) -> Dict[str, Any]:
        raise NotImplementedError()
