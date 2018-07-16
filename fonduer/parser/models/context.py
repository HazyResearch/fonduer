from sqlalchemy import Column, Integer, String

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
