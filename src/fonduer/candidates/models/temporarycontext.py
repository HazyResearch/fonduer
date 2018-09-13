from builtins import object

from sqlalchemy.sql import select, text

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
        if self.id is None:
            stable_id = self.get_stable_id()
            id = session.execute(
                select([Context.id]).where(Context.stable_id == stable_id)
            ).first()
            if id is None:
                self.id = session.execute(
                    Context.__table__.insert(),
                    {"type": self._get_table_name(), "stable_id": stable_id},
                ).inserted_primary_key[0]
                insert_args = self._get_insert_args()
                insert_args["id"] = self.id
                for (key, val) in insert_args.items():
                    if isinstance(val, list):
                        insert_args[key] = val
                session.execute(text(self._get_insert_query()), insert_args)
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

    def _get_table_name(self):
        raise NotImplementedError()

    def _get_insert_query(self):
        raise NotImplementedError()

    def _get_insert_args(self):
        raise NotImplementedError()
