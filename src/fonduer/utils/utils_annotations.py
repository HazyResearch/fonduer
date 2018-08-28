import logging
import subprocess
import tempfile
from builtins import range, str, zip
from collections import namedtuple

import numpy as np
import scipy.sparse as sparse
from pandas import DataFrame, Series

from fonduer.candidates.models import Candidate
from fonduer.meta import Meta
from fonduer.utils.utils import (
    matrix_conflicts,
    matrix_coverage,
    matrix_fn,
    matrix_fp,
    matrix_overlaps,
    matrix_tn,
    matrix_tp,
)

# Used to conform to existing annotation key API call
# Note that this annotation matrix class cannot be replaced with snorkel one
# since we do not have ORM-backed key objects but rather a simple python list.
_TempKey = namedtuple("TempKey", ["id", "name"])

# Grab a pointer to the global vars
_meta = Meta.init()

logger = logging.getLogger(__name__)
segment_dir = tempfile.gettempdir()


class csr_AnnotationMatrix(sparse.csr_matrix):
    """
    An extension of the scipy.sparse.csr_matrix class for holding sparse
    annotation matrices and related helper methods.
    """

    def __init__(self, arg1, **kwargs):
        # Map candidate id to row id
        self.candidate_index = kwargs.pop("candidate_index", {})
        # Map row id to candidate id
        self.row_index = kwargs.pop("row_index", [])
        # Map col id to key str
        self.keys = kwargs.pop("keys", [])
        # Map key str to col number
        self.key_index = kwargs.pop("key_index", {})

        # Note that scipy relies on the first three letters of the class to
        # define matrix type...
        super(csr_AnnotationMatrix, self).__init__(arg1, **kwargs)

    def get_candidate(self, session, i):
        """Return the Candidate object corresponding to row i"""
        return session.query(Candidate).filter(Candidate.id == self.row_index[i]).one()

    def get_row_index(self, candidate):
        """Return the row index of the Candidate"""
        return self.candidate_index[candidate.id]

    def get_key(self, j):
        """Return the AnnotationKey object corresponding to column j"""
        return _TempKey(j, self.keys[j])

    def get_col_index(self, key):
        """Return the cow index of the AnnotationKey"""
        return self.key_index[key.id]

    def stats(self):
        """Return summary stats about the annotations"""
        raise NotImplementedError()

    def lf_stats(self, labels=None, est_accs=None):
        """Returns a pandas DataFrame with the LFs and various per-LF statistics"""
        lf_names = self.keys

        # Default LF stats
        col_names = ["j", "Coverage", "Overlaps", "Conflicts"]
        d = {
            "j": list(range(self.shape[1])),
            "Coverage": Series(data=matrix_coverage(self), index=lf_names),
            "Overlaps": Series(data=matrix_overlaps(self), index=lf_names),
            "Conflicts": Series(data=matrix_conflicts(self), index=lf_names),
        }
        if labels is not None:
            col_names.extend(["TP", "FP", "FN", "TN", "Empirical Acc."])
            ls = np.ravel(labels.todense() if sparse.issparse(labels) else labels)
            tp = matrix_tp(self, ls)
            fp = matrix_fp(self, ls)
            fn = matrix_fn(self, ls)
            tn = matrix_tn(self, ls)
            ac = (tp + tn) / (tp + tn + fp + fn)
            d["Empirical Acc."] = Series(data=ac, index=lf_names)
            d["TP"] = Series(data=tp, index=lf_names)
            d["FP"] = Series(data=fp, index=lf_names)
            d["FN"] = Series(data=fn, index=lf_names)
            d["TN"] = Series(data=tn, index=lf_names)

        if est_accs is not None:
            col_names.append("Learned Acc.")
            d["Learned Acc."] = Series(data=est_accs, index=lf_names)
        return DataFrame(data=d, index=lf_names)[col_names]


def get_sql_name(text):
    """
    Create valid SQL identifier as part of a feature storage table name
    """
    # Normalize identifier
    text = "".join(c.lower() if c.isalnum() else " " for c in text)
    text = "_".join(text.split())
    return text


def tsv_escape(s):
    if s is None:
        return "\\N"
    # Make sure feature names are still uniquely encoded in ascii
    s = str(s)
    s = s.replace('"', '\\\\"').replace("\t", "\\t")
    if any(c in ",{}" for c in s):
        s = '"' + s + '"'
    return s


def array_tsv_escape(vals):
    return "{" + ",".join(tsv_escape(p) for p in vals) + "}"


def table_exists(con, name):
    cur = con.execute(
        "select exists(select * from information_schema.tables where table_name=%s)",
        (name,),
    )
    return cur.fetchone()[0]


def copy_postgres(segment_file_blob, table_name, tsv_columns):
    """
    @var segment_file_blob: e.g. "segment_*.tsv"
    @var table_name: The SQL table name to copy into
    @var tsv_columns: a string listing column names in the segment files
    separated by comma. e.g. "name, age, income"
    """
    logger.info("Copying {} to postgres".format(table_name))

    username = "-U " + _meta.DBUSER if _meta.DBUSER is not None else ""
    password = "PGPASSWORD=" + _meta.DBPWD if _meta.DBPWD is not None else ""
    port = "-p " + str(_meta.DBPORT) if _meta.DBPORT is not None else ""
    cmd = (
        'cat %s | %s psql %s %s %s -c "COPY %s(%s) '
        'FROM STDIN" --set=ON_ERROR_STOP=true'
    ) % (
        segment_file_blob,
        password,
        _meta.DBNAME,
        username,
        port,
        table_name,
        tsv_columns,
    )
    logger.debug(cmd)
    _out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    logger.info(_out)


def load_annotation_matrix(
    con,
    candidates,
    split,
    table_name,
    key_table_name,
    replace_key_set,
    storage,
    update_keys,
    ignore_keys,
):
    """
    Loads a sparse matrix from an annotation table
    """
    if replace_key_set:
        # Recalculate unique keys for this set of candidates
        con.execute("DROP TABLE IF EXISTS %s" % key_table_name)
    if replace_key_set or not table_exists(con, key_table_name):
        if storage == "COO":
            con.execute(
                "CREATE TABLE %s AS "
                "(SELECT DISTINCT key FROM %s)" % (key_table_name, table_name)
            )
        else:
            con.execute(
                "CREATE TABLE %s AS "
                "(SELECT DISTINCT UNNEST(keys) as key FROM %s)"
                % (key_table_name, table_name)
            )
        con.execute("ALTER TABLE %s ADD PRIMARY KEY(key)" % key_table_name)
    elif update_keys:
        if storage == "COO":
            con.execute(
                "INSERT INTO %s SELECT DISTINCT key FROM %s "
                "ON CONFLICT(key) DO NOTHING" % (key_table_name, table_name)
            )
        else:
            con.execute(
                "INSERT INTO %s SELECT DISTINCT UNNEST(keys) as key FROM %s "
                "ON CONFLICT(key) DO NOTHING" % (key_table_name, table_name)
            )

    # The result should be a list of all feature strings, small enough to hold
    # in memory
    # TODO: store the actual index in table in case row number is unstable
    # between queries
    ignore_keys = set(ignore_keys)
    keys = [
        row[0]
        for row in con.execute("SELECT * FROM %s" % key_table_name)
        if row[0] not in ignore_keys
    ]
    key_index = {key: i for i, key in enumerate(keys)}
    # Create sparse matrix in LIL format for incremental construction
    lil_feat_matrix = sparse.lil_matrix((len(candidates), len(keys)), dtype=np.int64)

    row_index = []
    candidate_index = {}
    # Load annotations from database
    # TODO: move this for-loop computation to database for automatic
    # parallelization, avoid communication overhead etc. Try to avoid the log
    # sorting factor using unnest
    if storage == "COO":
        logger.info("key size: {}".format(len(keys)))
        logger.info("candidate size {}".format(len(candidates)))
        iterator_sql = "SELECT candidate_id, key, value FROM %s "
        "WHERE candidate_id IN "
        "(SELECT id FROM candidate WHERE split=%d) "
        "ORDER BY candidate_id" % (table_name, split)
        prev_id = None
        i = -1
        for _, (candidate_id, key, value) in enumerate(con.execute(iterator_sql)):
            # Update candidate index tracker
            if candidate_id != prev_id:
                i += 1
                candidate_index[candidate_id] = i
                row_index.append(candidate_id)
                prev_id = candidate_id
            # Only keep known features
            key_id = key_index.get(key, None)
            if key_id is not None:
                lil_feat_matrix[i, key_id] = int(value)

    else:
        iterator_sql = """SELECT candidate_id, keys, values FROM %s
                          WHERE candidate_id IN
                          (SELECT id FROM candidate WHERE split=%d)
                          ORDER BY candidate_id""" % (
            table_name,
            split,
        )
        for i, (candidate_id, c_keys, values) in enumerate(con.execute(iterator_sql)):
            candidate_index[candidate_id] = i
            row_index.append(candidate_id)
            for key, value in zip(c_keys, values):
                # Only keep known features
                key_id = key_index.get(key, None)
                if key_id is not None:
                    lil_feat_matrix[i, key_id] = int(value)

    return csr_AnnotationMatrix(
        lil_feat_matrix,
        candidate_index=candidate_index,
        row_index=row_index,
        keys=keys,
        key_index=key_index,
    )
