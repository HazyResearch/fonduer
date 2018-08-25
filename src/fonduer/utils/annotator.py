import codecs
import logging
import os
import tempfile
from builtins import range, str, zip
from collections import namedtuple

from fonduer.candidates.models import Candidate
from fonduer.meta import Meta, new_sessionmaker
from fonduer.utils.udf import UDF, UDFRunner
from fonduer.utils.utils import remove_files
from fonduer.utils.utils_annotations import (
    array_tsv_escape,
    copy_postgres,
    get_sql_name,
    load_annotation_matrix,
    table_exists,
)

# Used to conform to existing annotation key API call
# Note that this annotation matrix class cannot be replaced with snorkel one
# since we do not have ORM-backed key objects but rather a simple python list.
_TempKey = namedtuple("TempKey", ["id", "name"])

# Grab a pointer to the global vars
_meta = Meta.init()

logger = logging.getLogger(__name__)
segment_dir = tempfile.gettempdir()


def _to_annotation_generator(fns):
    """"
    Generic method which takes a set of functions, and returns a generator that
    yields function.__name__, function result pairs.
    """

    def fn_gen(c):
        for f in fns:
            yield f.__name__, f(c)

    return fn_gen


def _segment_filename(db_name, table_name, job_id, start=None, end=None):
    suffix = "*"
    if start is not None:
        suffix = str(start)
        if end is not None:
            suffix += "-" + str(end)
    return "%s_%s_%s_%s.tsv" % (db_name, table_name, job_id, suffix)


class AnnotatorUDF(UDF):
    def __init__(self, f, **kwargs):
        self.anno_generator = (
            _to_annotation_generator(f) if hasattr(f, "__iter__") else f
        )
        super(AnnotatorUDF, self).__init__(**kwargs)

    def apply(self, batch_range, table_name, split, cache, **kwargs):
        """
        Applies a given function to a range of candidates

        Note: Accepts a id_range as argument, because of issues with putting
        Candidate subclasses into Queues (can't pickle...)
        """
        start, end = batch_range
        file_name = _segment_filename(_meta.DBNAME, table_name, split, self.worker_id)
        segment_path = os.path.join(segment_dir, file_name)
        candidates = (
            self.session.query(Candidate)
            .filter(Candidate.split == split)
            .order_by(Candidate.id)
            .slice(start, end)
        )
        with codecs.open(segment_path, "a+", encoding="utf-8") as writer:
            if not cache:
                for i, candidate in enumerate(candidates):
                    # Runs the actual extraction function
                    nonzero_kvs = [
                        (k, v) for k, v in self.anno_generator(candidate) if v != 0
                    ]
                    if nonzero_kvs:
                        keys, values = list(zip(*nonzero_kvs))
                    else:
                        keys = values = []
                    row = [
                        str(candidate.id),
                        array_tsv_escape(keys),
                        array_tsv_escape(values),
                    ]
                    writer.write("\t".join(row) + "\n")
            else:
                nonzero_kv_dict = {}
                for id, k, v in self.anno_generator(list(candidates)):
                    if id not in nonzero_kv_dict:
                        nonzero_kv_dict[id] = []
                    if v != 0:
                        nonzero_kv_dict[id].append((k, v))
                for i, candidate in enumerate(candidates):
                    nonzero_kvs = nonzero_kv_dict[candidate.id]
                    if nonzero_kvs:
                        keys, values = list(zip(*nonzero_kvs))
                    else:
                        keys = values = []
                    row = [
                        str(candidate.id),
                        array_tsv_escape(keys),
                        array_tsv_escape(values),
                    ]
                    writer.write("\t".join(row) + "\n")
        # This return + yield combination results in a purely empty generator
        # function. Specifically, the yield turns the function into a generator,
        # and the return terminates the generator before yielding anything.
        return
        yield


class Annotator(UDFRunner):
    """Abstract class for annotating candidates and persisting these
    annotations to DB.
    """

    def __init__(self, candidate_type, annotation_type, f, batch_size=50, **kwargs):
        self.candidate_type = candidate_type
        if isinstance(candidate_type, type):
            candidate_type = candidate_type.__name__
        self.table_name = get_sql_name(candidate_type) + "_" + annotation_type
        self.key_table_name = self.table_name + "_keys"
        self.annotation_type = annotation_type
        self.batch_size = batch_size
        super(Annotator, self).__init__(AnnotatorUDF, f=f, **kwargs)

    def apply(
        self,
        split,
        key_group=0,
        replace_key_set=True,
        update_keys=False,
        update_values=True,
        storage=None,
        ignore_keys=[],
        **kwargs
    ):
        if update_keys:
            replace_key_set = False
        # Get the cids based on the split, and also the count
        Session = new_sessionmaker()
        session = Session()

        # NOTE: In the current UDFRunner implementation, we load all these into
        # memory and fill a multiprocessing JoinableQueue with them before
        # starting... so might as well load them here and pass in. Also, if we
        # try to pass in a query iterator instead, with AUTOCOMMIT on, we get a
        # TXN error...
        candidates = (
            session.query(Candidate)
            .filter(Candidate.type == self.candidate_type.__tablename__)
            .filter(Candidate.split == split)
            .all()
        )
        cids_count = len(candidates)
        if cids_count == 0:
            raise ValueError("No candidates in current split")

        # Setting up job batches
        chunks = cids_count // self.batch_size
        batch_range = [
            (i * self.batch_size, (i + 1) * self.batch_size) for i in range(chunks)
        ]
        remainder = cids_count % self.batch_size
        if remainder:
            batch_range.append((chunks * self.batch_size, cids_count))

        old_table_name = None
        table_name = self.table_name
        # Run the Annotator
        with _meta.engine.connect() as con:
            table_already_exists = table_exists(con, table_name)
            if update_values and table_already_exists:
                # Now we extract under a temporary name for merging
                old_table_name = table_name
                table_name += "_updates"

            segment_file_blob = os.path.join(
                segment_dir, _segment_filename(_meta.DBNAME, self.table_name, split)
            )
            remove_files(segment_file_blob)
            cache = True if self.annotation_type == "feature" else False
            super(Annotator, self).apply(
                batch_range,
                table_name=self.table_name,
                split=split,
                cache=cache,
                **kwargs
            )

            # Insert and update keys
            if not table_already_exists or old_table_name:
                con.execute(
                    "CREATE TABLE {}(candidate_id integer PRIMARY KEY, "
                    "keys text[] NOT NULL, values real[] NOT NULL)".format(table_name)
                )
            copy_postgres(segment_file_blob, table_name, "candidate_id, keys, values")
            remove_files(segment_file_blob)

            # Replace the LIL table with COO if requested
            if storage == "COO":
                temp_coo_table = table_name + "_COO"
                con.execute(
                    "CREATE TABLE %s AS "
                    "(SELECT candidate_id, UNNEST(keys) as key, "
                    "UNNEST(values) as value from %s)" % (temp_coo_table, table_name)
                )
                con.execute("DROP TABLE %s" % table_name)
                con.execute(
                    "ALTER TABLE %s RENAME TO %s" % (temp_coo_table, table_name)
                )
                con.execute(
                    "ALTER TABLE %s ADD PRIMARY KEY(candidate_id, key)" % table_name
                )
                # Update old table
                if old_table_name:
                    con.execute(
                        "INSERT INTO %s SELECT * FROM %s "
                        "ON CONFLICT(candidate_id, key) "
                        "DO UPDATE SET value=EXCLUDED.value"
                        % (old_table_name, table_name)
                    )
                    con.execute("DROP TABLE %s" % table_name)
            else:  # LIL
                # Update old table
                if old_table_name:
                    con.execute(
                        "INSERT INTO %s AS old SELECT * FROM %s "
                        "ON CONFLICT(candidate_id) "
                        "DO UPDATE SET "
                        "values=old.values || EXCLUDED.values,"
                        "keys=old.keys || EXCLUDED.keys" % (old_table_name, table_name)
                    )
                    con.execute("DROP TABLE %s" % table_name)

            if old_table_name:
                table_name = old_table_name
            # Load the matrix
            key_table_name = self.key_table_name
            if key_group:
                key_table_name = self.key_table_name + "_" + get_sql_name(key_group)

            return load_annotation_matrix(
                con,
                candidates,
                split,
                table_name,
                key_table_name,
                replace_key_set,
                storage,
                update_keys,
                ignore_keys,
            )

    def clear(self, session, split, replace_key_set=False, **kwargs):
        """
        Deletes the Annotations for the Candidates in the given split.

        If replace_key_set=True, deletes *all* Annotations (of this Annotation
        sub-class) and also deletes all AnnotationKeys (of this sub-class)
        """
        with _meta.engine.connect() as con:
            if split is None:
                con.execute("DROP TABLE IF EXISTS %s" % self.table_name)
            elif table_exists(con, self.table_name):
                con.execute(
                    "DELETE FROM %s WHERE candidate_id IN "
                    "(SELECT id FROM candidate WHERE split=%d)"
                    % (self.table_name, split)
                )
            if replace_key_set:
                con.execute("DROP TABLE IF EXISTS %s" % self.key_table_name)

    def apply_existing(self, split, key_group=0, **kwargs):
        """Alias for apply that emphasizes we are using an existing AnnotatorKey set."""
        return self.apply(split, key_group=key_group, replace_key_set=False, **kwargs)

    def load_matrix(self, split, ignore_keys=[]):
        Session = new_sessionmaker()
        session = Session()
        candidates = session.query(Candidate).filter(Candidate.split == split).all()
        with _meta.engine.connect() as con:
            return load_annotation_matrix(
                con,
                candidates,
                split,
                self.table_name,
                self.key_table_name,
                False,
                None,
                False,
                ignore_keys,
            )
