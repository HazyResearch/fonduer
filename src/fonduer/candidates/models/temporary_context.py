from builtins import object

from sqlalchemy.sql import select

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

    def __init__(self):
        self.id = None

    def _load_id_or_insert(self, session):
        """Load the id of the temporary context if it exists or return insert args.

        As a side effect, this also inserts the Context object for the stableid.

        :return: The record of the temporary context to insert.
        :rtype: dict
        """
        if self.id is None:
            stable_id = self.get_stable_id()
            # Check if exists
            id = session.execute(
                select([Context.id]).where(Context.stable_id == stable_id)
            ).first()

            # If not, insert
            if id is None:
                self.id = session.execute(
                    Context.__table__.insert(),
                    {"type": self._get_table().__tablename__, "stable_id": stable_id},
                ).inserted_primary_key[0]
                insert_args = self._get_insert_args()
                insert_args["id"] = self.id
                return insert_args
            else:
                self.id = id[0]

    def __eq__(self, other):
        raise NotImplementedError()

    def __ne__(self, other):
        raise NotImplementedError()

    def __hash__(self):
        raise NotImplementedError()

    def _get_polymorphic_identity(self):
        raise NotImplementedError()

    def get_stable_id(self):
        raise NotImplementedError()

    def _get_table(self):
        raise NotImplementedError()

    def _get_insert_args(self):
        raise NotImplementedError()
