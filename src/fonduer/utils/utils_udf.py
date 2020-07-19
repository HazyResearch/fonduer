"""Fonduer UDF utils."""
import logging
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import numpy as np
from scipy.sparse import csr_matrix
from sqlalchemy import String, Table
from sqlalchemy.dialects.postgresql import ARRAY, insert
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import cast

from fonduer.candidates.models import Candidate
from fonduer.parser.models import Document
from fonduer.utils.models import AnnotationMixin

logger = logging.getLogger(__name__)

# Flag value to signal that no filtering on split should be applied. Not an
# integer to ensure that it won't conflict with a user's split value.
ALL_SPLITS = "ALL"


def _get_cand_values(candidate: Candidate, key_table: Table) -> List[AnnotationMixin]:
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
        raise ValueError(f"{key_table} is not a valid key table.")


def _batch_postgres_query(
    table: Table, records: List[Dict[str, Any]]
) -> Iterator[List[Dict[str, Any]]]:
    """Break the list into chunks that can be processed as a single statement.

    Postgres query cannot be too long or it will fail.
    See: https://dba.stackexchange.com/questions/131399/is-there-a-maximum-
     length-constraint-for-a-postgres-query

    :param records: The full list of records to batch.
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
            logger.debug(f"Splitting query due to length ({total_len} chars).")
            yield records[start:end]
            start = end
            # Reset the total query length
            total_len = len(preamble)
        else:
            total_len += record_len

    yield records[start:end]


def get_sparse_matrix_keys(session: Session, key_table: Table) -> List:
    """Return a list of keys for the sparse matrix."""
    return session.query(key_table).order_by(key_table.name).all()


def batch_upsert_records(
    session: Session, table: Table, records: List[Dict[str, Any]]
) -> None:
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


def get_sparse_matrix(
    session: Session,
    key_table: Table,
    cand_lists: Union[Sequence[Candidate], Iterable[Sequence[Candidate]]],
    key: Optional[str] = None,
) -> List[csr_matrix]:
    """Load sparse matrix of GoldLabels for each candidate_class."""
    result = []
    cand_lists = cand_lists if isinstance(cand_lists, (list, tuple)) else [cand_lists]

    for cand_list in cand_lists:
        if len(cand_list) == 0:
            raise ValueError("cand_lists contain empty cand_list.")

        # Keys are used as a global index
        if key:
            key_names = [key]
        else:
            # Get all keys
            all_keys = get_sparse_matrix_keys(session, key_table)
            # Filter only keys that are used by this cand_list
            key_names = [k.name for k in all_keys]

        annotations: List[Dict[str, Any]] = []
        for cand in cand_list:
            annotation_mixins: List[AnnotationMixin] = _get_cand_values(cand, key_table)
            if annotation_mixins:
                annotations.append(
                    {
                        "keys": annotation_mixins[0].keys,
                        "values": annotation_mixins[0].values,
                    }
                )
            else:
                annotations.append({"keys": [], "values": []})
        result.append(_convert_mappings_to_matrix(annotations, key_names))
    return result


def _convert_mappings_to_matrix(
    mappings: List[Dict[str, Any]], keys: List[str]
) -> csr_matrix:
    """Convert a list of (annotation) mapping into a sparse matrix.

    An annotation mapping is a dictionary representation of annotations like instances
    of :class:`Label` and :class:`Feature`. For example, label.keys and label.values
    corresponds to annotation["keys"] and annotation["values"].

    Note that :func:`FeaturizerUDF.apply` returns a list of list of such a mapping,
    where the outer list represents candidate_classes, while this method takes a list
    of a mapping of each candidate_class.

    :param mappings: a list of annotation mapping.
    :param keys: a list of keys, which becomes columns of the matrix to be returned.
    """
    # Create a mapping that maps key_name to column index)
    keys_map = {key: keys.index(key) for key in keys}

    indptr = [0]
    indices = []
    data = []
    for mapping in mappings:
        if mapping:
            for key, value in zip(mapping["keys"], mapping["values"]):
                if key in keys_map:
                    indices.append(keys_map[key])
                    data.append(value)
        indptr.append(len(indices))
    return csr_matrix((data, indices, indptr), shape=(len(mappings), len(keys)))


def unshift_label_matrix(L_sparse: csr_matrix) -> np.ndarray:
    """Unshift a sparse label matrix (ABSTAIN as 0) to a dense one (ABSTAIN as -1)."""
    return L_sparse.toarray() - 1


def shift_label_matrix(L: np.ndarray) -> csr_matrix:
    """Shift a dense label matrix (ABSTAIN as -1) to a sparse one (ABSTAIN as 0)."""
    return csr_matrix(L + 1)


def get_docs_from_split(
    session: Session, candidate_classes: Iterable[Type[Candidate]], split: int
) -> Set[Document]:
    """Return a list of documents that contain the candidates in the split."""
    # Only grab the docs containing candidates from the given split.
    sub_query = session.query(Candidate.id).filter(Candidate.split == split).subquery()
    split_docs: Set[Document] = set()
    for candidate_class in candidate_classes:
        split_docs.update(
            cand.document
            for cand in session.query(candidate_class)
            .filter(candidate_class.id.in_(sub_query))
            .all()
        )
    return split_docs


def get_mapping(
    table: Table,
    candidates: Iterable[Candidate],
    generator: Callable[[Candidate], Iterator[Tuple]],
) -> Iterator[Dict[str, Any]]:
    """Generate map of keys and values for the candidate from the generator.

    :param table: The table we will be inserting into (i.e. Feature or Label).
    :param candidates: The candidates to get mappings for.
    :param generator: A generator yielding (candidate_id, key, value) tuples.
    :return: Generator of dictionaries of {"candidate_id": _, "keys": _, "values": _}
    """
    for cand in candidates:
        # Grab the old values
        if len(getattr(cand, table.__tablename__ + "s")) != 0:
            temp = getattr(cand, table.__tablename__ + "s")[0]
            cand_map = dict(zip(temp.keys, temp.values))
        else:
            cand_map = {}

        for cid, key, value in generator(cand):
            if value == 0:
                # Make sure this key does not exist in cand_map
                cand_map.pop(key, None)
                continue
            cand_map[key] = value

        # Assemble label arguments
        yield {
            "candidate_id": cand.id,
            "keys": [*cand_map.keys()],
            "values": [*cand_map.values()],
        }


def drop_all_keys(
    session: Session, key_table: Table, candidate_classes: Iterable[Type[Candidate]]
) -> None:
    """Bulk drop annotation keys for all the candidate_classes in the table.

    Rather than directly dropping the keys, this removes the candidate_classes
    specified for the given keys only. If all candidate_classes are removed for
    a key, the key is dropped.

    :param key_table: The sqlalchemy class to insert into.
    :param candidate_classes: A list of candidate classes to drop.
    """
    if not candidate_classes:
        return

    set_of_candidate_classes: Set[str] = set(
        [c.__tablename__ for c in candidate_classes]
    )

    # Select all rows that contain ANY of the candidate_classes
    all_rows = (
        session.query(key_table)
        .filter(
            key_table.candidate_classes.overlap(
                cast(set_of_candidate_classes, ARRAY(String))
            )
        )
        .all()
    )
    to_delete = set()
    to_update = []

    # All candidate classes will be the same for all keys, so just look at one
    for row in all_rows:
        # Remove the selected candidate_classes. If empty, mark for deletion.
        row.candidate_classes = list(
            set(row.candidate_classes) - set_of_candidate_classes
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


def drop_keys(session: Session, key_table: Table, keys: Dict) -> None:
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


def upsert_keys(session: Session, key_table: Table, keys: Dict) -> None:
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
                logger.debug(e)
