import logging

from scipy.sparse import csr_matrix

from fonduer import Meta
from fonduer.candidates.models import Candidate

logger = logging.getLogger(__name__)

# Flag value to signal that no filtering on split should be applied. Not an
# integer to ensure that it won't conflict with a user's split value.
ALL_SPLITS = "ALL"


def _get_cand_values(candidate, key_table):
    """Get the corresponding values for the key_table."""
    # NOTE: Import just before checking to avoid circular imports.
    from fonduer.features.models import FeatureKey
    from fonduer.supervision.models import GoldLabelKey, LabelKey

    if key_table == FeatureKey:
        return candidate.features
    elif key_table == LabelKey:
        return candidate.labels
    elif key_table == GoldLabelKey:
        return candidate.gold_labels
    else:
        raise ValueError("{} is not a valid key table.".format(key_table))


def get_sparse_matrix(session, key_table, cand_lists, key=None):
    """Load sparse matrix of GoldLabels for each candidate_class."""
    result = []
    cand_lists = cand_lists if isinstance(cand_lists, (list, tuple)) else [cand_lists]
    for cand_list in cand_lists:
        if key:
            keys = [key]
        else:
            keys = [
                key.name
                for key in session.query(key_table).order_by(key_table.name).all()
            ]

        indptr = [0]
        indices = []
        data = []
        for cand in cand_list:
            values = _get_cand_values(cand, key_table)
            if values:
                cand_keys = values[0].keys
                cand_values = values[0].values
                indices.extend([keys.index(key) for key in cand_keys if key in keys])
                data.extend(
                    [cand_values[i[0]] for i in enumerate(cand_keys) if i[1] in keys]
                )

            indptr.append(len(indices))

        result.append(csr_matrix((data, indices, indptr)))

    return result


def docs_from_split(session, candidate_classes, split):
    """Return a list of documents that contain the candidates in the split."""
    # Only grab the docs containing candidates from the given split.
    sub_query = session.query(Candidate.id).filter(Candidate.split == split).subquery()
    split_docs = set()
    for candidate_class in candidate_classes:
        split_docs.update(
            cand.document
            for cand in session.query(candidate_class)
            .filter(candidate_class.id.in_(sub_query))
            .all()
        )
    return split_docs


def get_mapping(candidates, generator, key_set):
    """Generate map of keys and values for the candidate from the generator.

    :param key_set: A mutable set which keys will be added to.
    """
    for cand in candidates:
        map_args = {"candidate_id": cand.id}
        keys = []
        values = []
        for cid, key, value in generator(cand):
            if value == 0:
                continue
            keys.append(key)
            values.append(value)

        # Assemble label arguments
        map_args["keys"] = keys
        map_args["values"] = values

        # mutate the passed in key_set
        key_set.update(keys)
        yield map_args


def cands_from_split(session, candidate_classes, doc, split):
    """Return the candidates from this document based on the split."""
    cands = []
    if split == ALL_SPLITS:
        # Get cands from all splits
        for candidate_class in candidate_classes:
            cands.extend(
                session.query(candidate_class)
                .filter(candidate_class.document_id == doc.id)
                .all()
            )
    else:
        # Get cands from the specified split
        for candidate_class in candidate_classes:
            cands.extend(
                session.query(candidate_class)
                .filter(candidate_class.document_id == doc.id)
                .filter(candidate_class.split == split)
                .all()
            )
    return cands


def add_keys(session, key_table, keys):
    """Bulk add annotation keys to the specified table.

    :param table: The sqlalchemy class to insert into.
    :param keys: A list of strings to insert into the table.
    """
    # Do nothing if empty
    if not keys:
        logger.warning(
            "Attempted to insert empty list of keys to {}.".format(
                key_table.__tablename__
            )
        )
        return

    # NOTE: If you pprint these values, it may look funny because of the
    # newlines and tabs as whitespace characters in these names. Use normal
    # print.
    existing_keys = [k.name for k in session.query(key_table).all()]
    new_keys = [k for k in keys if k not in existing_keys]

    # Bulk insert all new feature keys
    if new_keys:
        Meta.engine.execute(
            key_table.__table__.insert(), [{"name": key} for key in new_keys]
        )
