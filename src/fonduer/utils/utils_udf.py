import logging

from scipy.sparse import csr_matrix
from sqlalchemy import String
from sqlalchemy.dialects.postgresql import ARRAY, insert
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.sql.expression import cast

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


def _batch_postgres_query(table, records):
    """Break the list into chunks that can be processed as a single statement.

    Postgres query cannot be too long or it will fail.
    See: https://dba.stackexchange.com/questions/131399/is-there-a-maximum-
     length-constraint-for-a-postgres-query

    :param records: The full list of records to batch.
    :type records: iterable
    :param table: The sqlalchemy table.
    :return: A generator of lists of records.
    """
    if not records:
        return

    POSTGRESQL_MAX = 0x3FFFFFFF

    # Create preamble and measure its length
    preamble = (
        "INSERT INTO "
        + table.__tablename__
        + " ("
        + ", ".join(records[0].keys())
        + ") VALUES ("
        + ", ".join(["?"] * len(records[0].keys()))
        + ")\n"
    )
    start = 0
    end = 0
    total_len = len(preamble)
    while end < len(records):
        record_len = sum([len(str(v)) for v in records[end].values()])

        # Pre-increment to include the end element in the slice
        end += 1

        if total_len + record_len >= POSTGRESQL_MAX:
            logger.debug("Splitting query due to length ({} chars).".format(total_len))
            yield records[start:end]
            start = end
            # Reset the total query length
            total_len = len(preamble)
        else:
            total_len += record_len

    yield records[start:end]


def get_sparse_matrix_keys(session, key_table):
    """Return a list of keys for the sparse matrix."""
    return session.query(key_table).order_by(key_table.name).all()


def batch_upsert_records(session, table, records):
    """Batch upsert records into postgresql database."""
    if not records:
        return
    for record_batch in _batch_postgres_query(table, records):
        stmt = insert(table.__table__)
        stmt = stmt.on_conflict_do_update(
            constraint=table.__table__.primary_key,
            set_={
                "keys": stmt.excluded.get("keys"),
                "values": stmt.excluded.get("values"),
            },
        )
        session.execute(stmt, record_batch)
        session.commit()


def get_sparse_matrix(session, key_table, cand_lists, key=None):
    """Load sparse matrix of GoldLabels for each candidate_class."""
    result = []
    cand_lists = cand_lists if isinstance(cand_lists, (list, tuple)) else [cand_lists]

    for cand_list in cand_lists:
        if len(cand_list) == 0:
            raise ValueError("cand_lists contain empty cand_list.")
        candidate_class = cand_list[0].__tablename__

        # Keys are used as a global index
        if key:
            keys_map = {key: 0}
            key_size = len(keys_map)
        else:
            all_keys = get_sparse_matrix_keys(session, key_table)
            key_size = len(all_keys)
            keys_map = {}
            for (i, k) in enumerate(all_keys):
                if candidate_class in k.candidate_classes:
                    keys_map[k.name] = i

        indptr = [0]
        indices = []
        data = []
        for cand in cand_list:
            values = _get_cand_values(cand, key_table)
            if values:
                for cand_key, cand_value in zip(values[0].keys, values[0].values):
                    if cand_key in keys_map:
                        indices.append(keys_map[cand_key])
                        data.append(cand_value)

            indptr.append(len(indices))

        result.append(
            csr_matrix((data, indices, indptr), shape=(len(cand_list), key_size))
        )

    return result


def get_docs_from_split(session, candidate_classes, split):
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


def get_mapping(session, table, candidates, generator, key_map):
    """Generate map of keys and values for the candidate from the generator.

    :param session: The database session.
    :param table: The table we will be inserting into (i.e. Feature or Label).
    :param candidates: The candidates to get mappings for.
    :param generator: A generator yielding (candidate_id, key, value) tuples.
    :param key_map: A mutable dict which values will be added to as {key:
        [relations]}.
    :type key_map: Dict
    :return: Generator of dictionaries of {"candidate_id": _, "keys": _, "values": _}
    :rtype: generator of dict
    """
    for cand in candidates:
        # Grab the old values currently in the DB
        try:
            temp = session.query(table).filter(table.candidate_id == cand.id).one()
            cand_map = dict(zip(temp.keys, temp.values))
        except NoResultFound:
            cand_map = {}

        map_args = {"candidate_id": cand.id}
        for cid, key, value in generator(cand):
            if value == 0:
                continue
            cand_map[key] = value

        # Assemble label arguments
        map_args["keys"] = [*cand_map.keys()]
        map_args["values"] = [*cand_map.values()]

        # Update key_map by adding the candidate class for each key
        for key in map_args["keys"]:
            try:
                key_map[key].add(cand.__class__.__tablename__)
            except KeyError:
                key_map[key] = {cand.__class__.__tablename__}
        yield map_args


def get_cands_list_from_split(session, candidate_classes, doc, split):
    """Return the list of list of candidates from this document based on the split."""
    cands = []
    if split == ALL_SPLITS:
        # Get cands from all splits
        for candidate_class in candidate_classes:
            cands.append(
                session.query(candidate_class)
                .filter(candidate_class.document_id == doc.id)
                .all()
            )
    else:
        # Get cands from the specified split
        for candidate_class in candidate_classes:
            cands.append(
                session.query(candidate_class)
                .filter(candidate_class.document_id == doc.id)
                .filter(candidate_class.split == split)
                .all()
            )
    return cands


def drop_all_keys(session, key_table, candidate_classes):
    """Bulk drop annotation keys for all the candidate_classes in the table.

    Rather than directly dropping the keys, this removes the candidate_classes
    specified for the given keys only. If all candidate_classes are removed for
    a key, the key is dropped.

    :param key_table: The sqlalchemy class to insert into.
    :param candidate_classes: A list of candidate classes to drop.
    """
    if not candidate_classes:
        return

    candidate_classes = set([c.__tablename__ for c in candidate_classes])

    # Select all rows that contain ANY of the candidate_classes
    all_rows = (
        session.query(key_table)
        .filter(
            key_table.candidate_classes.overlap(cast(candidate_classes, ARRAY(String)))
        )
        .all()
    )
    to_delete = set()
    to_update = []

    # All candidate classes will be the same for all keys, so just look at one
    for row in all_rows:
        # Remove the selected candidate_classes. If empty, mark for deletion.
        row.candidate_classes = list(
            set(row.candidate_classes) - set(candidate_classes)
        )
        if len(row.candidate_classes) == 0:
            to_delete.add(row.name)
        else:
            to_update.append(
                {"name": row.name, "candidate_classes": row.candidate_classes}
            )

    # Perform all deletes
    if to_delete:
        query = session.query(key_table).filter(key_table.name.in_(to_delete))
        query.delete(synchronize_session="fetch")

    # Perform all updates
    if to_update:
        for batch in _batch_postgres_query(key_table, to_update):
            stmt = insert(key_table.__table__)
            stmt = stmt.on_conflict_do_update(
                constraint=key_table.__table__.primary_key,
                set_={
                    "name": stmt.excluded.get("name"),
                    "candidate_classes": stmt.excluded.get("candidate_classes"),
                },
            )
            session.execute(stmt, batch)
            session.commit()


def drop_keys(session, key_table, keys):
    """Bulk drop annotation keys to the specified table.

    Rather than directly dropping the keys, this removes the candidate_classes
    specified for the given keys only. If all candidate_classes are removed for
    a key, the key is dropped.

    :param key_table: The sqlalchemy class to insert into.
    :param keys: A map of {name: [candidate_classes]}.
    """
    # Do nothing if empty
    if not keys:
        return

    for key_batch in _batch_postgres_query(
        key_table, [{"name": k[0], "candidate_classes": k[1]} for k in keys.items()]
    ):
        all_rows = (
            session.query(key_table)
            .filter(key_table.name.in_([key["name"] for key in key_batch]))
            .all()
        )

        to_delete = set()
        to_update = []

        # All candidate classes will be the same for all keys, so just look at one
        candidate_classes = key_batch[0]["candidate_classes"]
        for row in all_rows:
            # Remove the selected candidate_classes. If empty, mark for deletion.
            row.candidate_classes = list(
                set(row.candidate_classes) - set(candidate_classes)
            )
            if len(row.candidate_classes) == 0:
                to_delete.add(row.name)
            else:
                to_update.append(
                    {"name": row.name, "candidate_classes": row.candidate_classes}
                )

        # Perform all deletes
        if to_delete:
            query = session.query(key_table).filter(key_table.name.in_(to_delete))
            query.delete(synchronize_session="fetch")

        # Perform all updates
        if to_update:
            stmt = insert(key_table.__table__)
            stmt = stmt.on_conflict_do_update(
                constraint=key_table.__table__.primary_key,
                set_={
                    "name": stmt.excluded.get("name"),
                    "candidate_classes": stmt.excluded.get("candidate_classes"),
                },
            )
            session.execute(stmt, to_update)
            session.commit()


def upsert_keys(session, key_table, keys):
    """Bulk add annotation keys to the specified table.

    :param key_table: The sqlalchemy class to insert into.
    :param keys: A map of {name: [candidate_classes]}.
    """
    # Do nothing if empty
    if not keys:
        return

    for key_batch in _batch_postgres_query(
        key_table, [{"name": k[0], "candidate_classes": k[1]} for k in keys.items()]
    ):
        stmt = insert(key_table.__table__)
        stmt = stmt.on_conflict_do_update(
            constraint=key_table.__table__.primary_key,
            set_={
                "name": stmt.excluded.get("name"),
                "candidate_classes": stmt.excluded.get("candidate_classes"),
            },
        )
        while True:
            try:
                session.execute(stmt, key_batch)
                session.commit()
                break
            except Exception as e:
                logger.debug("{}".format(e))
