from builtins import object

from sqlalchemy import Column, Integer, String
from sqlalchemy.sql import select, text

from fonduer.meta import Meta

# Grab pointer to global metadata
_meta = Meta.init()


class Context(_meta.Base):
    """
    A piece of content from which Candidates are composed.
    """

    __tablename__ = "context"
    id = Column(Integer, primary_key=True)
    type = Column(String, nullable=False)
    stable_id = Column(String, unique=True, nullable=False)

    __mapper_args__ = {"polymorphic_identity": "context", "polymorphic_on": type}

    def get_parent(self):
        raise NotImplementedError()

    def get_children(self):
        raise NotImplementedError()

    def get_sentence_generator(self):
        raise NotImplementedError()


class TemporaryContext(object):
    """
    A context which does not incur the overhead of a proper ORM-based Context
    object. The TemporaryContext class is specifically for the candidate
    extraction process, during which a CandidateSpace object will generate many
    TemporaryContexts, which will then be filtered by Matchers prior to
    materialization of Candidates and constituent Context objects.

    Every Context object has a corresponding TemporaryContext object from which
    it inherits.

    A TemporaryContext must have specified equality / set membership semantics,
    a stable_id for checking uniqueness against the database, and a promote()
    method which returns a corresponding Context object.
    """

    def __init__(self):
        self.id = None

    def load_id_or_insert(self, session):
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


def construct_stable_id(
    parent_context,
    polymorphic_type,
    relative_char_offset_start,
    relative_char_offset_end,
):
    """Contruct a stable ID for a Context given its parent and its character offsets relative to the parent"""
    doc_id, _, parent_doc_char_start, _ = split_stable_id(parent_context.stable_id)
    start = parent_doc_char_start + relative_char_offset_start
    end = parent_doc_char_start + relative_char_offset_end
    return "{}::{}:{}:{}".format(doc_id, polymorphic_type, start, end)


def split_stable_id(stable_id):
    """Split stable id, returning:

        * Document (root) stable ID
        * Context polymorphic type
        * Character offset start, end *relative to document start*

    Returns tuple of four values.
    """
    split1 = stable_id.split("::")
    if len(split1) == 2:
        split2 = split1[1].split(":")
        if len(split2) == 3:
            return split1[0], split2[0], int(split2[1]), int(split2[2])
    raise ValueError("Malformed stable_id:", stable_id)
