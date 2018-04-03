from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging

import numpy as np
import scipy.sparse as sparse
from pandas import DataFrame, Series

from .models import Candidate, GoldLabel, GoldLabelKey, Marginal
from .utils import (matrix_conflicts, matrix_coverage, matrix_fn, matrix_fp,
                    matrix_overlaps, matrix_tn, matrix_tp)


class csr_AnnotationMatrix(sparse.csr_matrix):
    """
    An extension of the scipy.sparse.csr_matrix class for holding sparse annotation matrices
    and related helper methods.
    """

    def __init__(self, arg1, **kwargs):
        # Note: Currently these need to return None if unset, otherwise matrix copy operations break...
        self.candidate_index = kwargs.pop('candidate_index', None)
        self.row_index = kwargs.pop('row_index', None)
        self.annotation_key_cls = kwargs.pop('annotation_key_cls', None)
        self.key_index = kwargs.pop('key_index', None)
        self.col_index = kwargs.pop('col_index', None)

        # Note that scipy relies on the first three letters of the class to define matrix type...
        super(csr_AnnotationMatrix, self).__init__(arg1, **kwargs)

    def get_candidate(self, session, i):
        """Return the Candidate object corresponding to row i"""
        return session.query(Candidate).filter(
            Candidate.id == self.row_index[i]).one()

    def get_row_index(self, candidate):
        """Return the row index of the Candidate"""
        return self.candidate_index[candidate.id]

    def get_key(self, session, j):
        """Return the AnnotationKey object corresponding to column j"""
        return session.query(self.annotation_key_cls).filter(
            self.annotation_key_cls.id == self.col_index[j]).one()

    def get_col_index(self, key):
        """Return the cow index of the AnnotationKey"""
        return self.key_index[key.id]

    def _get_sliced_indexes(self, s, axis, index, inv_index):
        """
        Remaps the indexes between matrix rows/cols and candidates/keys.
        Note: This becomes a massive performance bottleneck if not implemented
        properly, so be careful of changing!
        """
        if isinstance(s, slice):
            # Check for empty slice
            if s.start is None and s.stop is None:
                return index, inv_index
            else:
                idxs = np.arange(self.shape[axis])[s]
        elif isinstance(s, int):
            idxs = np.array([s])
        else:  # s is an array of ints
            idxs = s
            # If s is the entire slice, skip the remapping step
            if np.array_equal(idxs, list(range(len(idxs)))):
                return index, inv_index

        index_new, inv_index_new = {}, {}
        for i_new, i in enumerate(idxs):
            k = index[i]
            index_new[i_new] = k
            inv_index_new[k] = i_new
        return index_new, inv_index_new

    def __getitem__(self, key):
        X = super(csr_AnnotationMatrix, self).__getitem__(key)

        # If X is an integer or float value, just return it
        if (type(X) in [int, float] or issubclass(type(X), np.integer)
                or issubclass(type(X), np.float)):
            return X
        # If X is a matrix, make sure it stays a csr_AnnotationMatrix
        elif not isinstance(X, csr_AnnotationMatrix):
            X = csr_AnnotationMatrix(X)
        # X must be a matrix, so update appropriate csr_AnnotationMatrix fields
        X.annotation_key_cls = self.annotation_key_cls
        row_slice, col_slice = self._unpack_index(key)
        X.row_index, X.candidate_index = self._get_sliced_indexes(
            row_slice, 0, self.row_index, self.candidate_index)
        X.col_index, X.key_index = self._get_sliced_indexes(
            col_slice, 1, self.col_index, self.key_index)
        return X

    def stats(self):
        """Return summary stats about the annotations"""
        raise NotImplementedError()


class csr_LabelMatrix(csr_AnnotationMatrix):
    def lf_stats(self, session, labels=None, est_accs=None):
        """Returns a pandas DataFrame with the LFs and various per-LF statistics"""
        lf_names = [
            self.get_key(session, j).name for j in range(self.shape[1])
        ]

        # Default LF stats
        col_names = ['j', 'Coverage', 'Overlaps', 'Conflicts']
        d = {
            'j': list(range(self.shape[1])),
            'Coverage': Series(data=matrix_coverage(self), index=lf_names),
            'Overlaps': Series(data=matrix_overlaps(self), index=lf_names),
            'Conflicts': Series(data=matrix_conflicts(self), index=lf_names)
        }
        if labels is not None:
            col_names.extend(['TP', 'FP', 'FN', 'TN', 'Empirical Acc.'])
            ls = np.ravel(labels.todense()
                          if sparse.issparse(labels) else labels)
            tp = matrix_tp(self, ls)
            fp = matrix_fp(self, ls)
            fn = matrix_fn(self, ls)
            tn = matrix_tn(self, ls)
            ac = (tp + tn) / (tp + tn + fp + fn)
            d['Empirical Acc.'] = Series(data=ac, index=lf_names)
            d['TP'] = Series(data=tp, index=lf_names)
            d['FP'] = Series(data=fp, index=lf_names)
            d['FN'] = Series(data=fn, index=lf_names)
            d['TN'] = Series(data=tn, index=lf_names)

        if est_accs is not None:
            col_names.append('Learned Acc.')
            d['Learned Acc.'] = est_accs
            d['Learned Acc.'].index = lf_names
        return DataFrame(data=d, index=lf_names)[col_names]


def load_matrix(matrix_class,
                annotation_key_class,
                annotation_class,
                session,
                split=0,
                cids_query=None,
                key_group=0,
                key_names=None,
                zero_one=False,
                load_as_array=False):
    """
    Returns the annotations corresponding to a split of candidates with N members
    and an AnnotationKey group with M distinct keys as an N x M CSR sparse matrix.
    """
    cid_query = cids_query or session.query(Candidate.id)\
                                     .filter(Candidate.split == split)
    cid_query = cid_query.order_by(Candidate.id)

    keys_query = session.query(annotation_key_class.id)
    keys_query = keys_query.filter(annotation_key_class.group == key_group)
    if key_names is not None:
        keys_query = keys_query.filter(
            annotation_key_class.name.in_(frozenset(key_names)))
    keys_query = keys_query.order_by(annotation_key_class.id)

    # First, we query to construct the row index map
    cid_to_row = {}
    row_to_cid = {}
    for cid, in cid_query.all():
        if cid not in cid_to_row:
            j = len(cid_to_row)

            # Create both mappings
            cid_to_row[cid] = j
            row_to_cid[j] = cid

    # Second, we query to construct the column index map
    kid_to_col = {}
    col_to_kid = {}
    for kid, in keys_query.all():
        if kid not in kid_to_col:
            j = len(kid_to_col)

            # Create both mappings
            kid_to_col[kid] = j
            col_to_kid[j] = kid

    # Create sparse matrix in COO format for incremental construction
    row = []
    columns = []
    data = []

    # Rely on the core for fast iteration
    annot_select_query = annotation_class.__table__.select()

    # Iteratively construct row index and output sparse matrix
    # Cycles through the entire table to load the data.
    # Performance may slow down based on table size; however, negligible since
    # it takes 8min to go throuh 245M rows (pretty fast).
    for res in session.execute(annot_select_query):
        # NOTE: The order of return seems to be switched in Python 3???
        # Either way, make sure the order is set here explicitly!
        cid, kid, val = res.candidate_id, res.key_id, res.value

        if cid in cid_to_row and kid in kid_to_col:

            # Optionally restricts val range to {0,1}, mapping -1 -> 0
            if zero_one:
                val = 1 if val == 1 else 0
            row.append(cid_to_row[cid])
            columns.append(kid_to_col[kid])
            data.append(int(val))

    X = sparse.coo_matrix(
        (data, (row, columns)), shape=(len(cid_to_row), len(kid_to_col)))

    # Return as an AnnotationMatrix
    Xr = matrix_class(
        X,
        candidate_index=cid_to_row,
        row_index=row_to_cid,
        annotation_key_cls=annotation_key_class,
        key_index=kid_to_col,
        col_index=col_to_kid)
    return np.squeeze(Xr.toarray()) if load_as_array else Xr


def load_gold_labels(session, annotator_name, **kwargs):
    return load_matrix(
        csr_LabelMatrix,
        GoldLabelKey,
        GoldLabel,
        session,
        key_names=[annotator_name],
        **kwargs)


def save_marginals(session, X, marginals, training=True):
    """Save marginal probabilities for a set of Candidates to db.

    :param X: Either an M x N csr_AnnotationMatrix-class matrix, where M
        is number of candidates, N number of LFs/features; OR a list of
        arbitrary objects with candidate ids accessible via a .id attrib
    :param marginals: A dense M x K matrix of marginal probabilities, where
        K is the cardinality of the candidates, OR a M-dim list/array if K=2.
    :param training: If True, these are training marginals / labels; else they
        are saved as end model predictions.

    Note: The marginals for k=0 are not stored, only for k = 1,...,K
    """
    logger = logging.getLogger(__name__)
    # Make sure that we are working with a numpy array
    try:
        shape = marginals.shape
    except Exception as e:
        marginals = np.array(marginals)
        shape = marginals.shape

    # Handle binary input as M x 1-dim array; assume elements represent
    # poksitive (k=1) class values
    if len(shape) == 1:
        marginals = np.vstack([1 - marginals, marginals]).T

    # Only add values for classes k=1,...,K
    marginal_tuples = []
    for i in range(shape[0]):
        for k in range(1, shape[1] if len(shape) > 1 else 2):
            if marginals[i, k] > 0:
                marginal_tuples.append((i, k, marginals[i, k]))

    # NOTE: This will delete all existing marginals of type `training`
    session.query(Marginal).filter(Marginal.training == training).\
        delete(synchronize_session='fetch')

    # Prepare bulk INSERT query
    q = Marginal.__table__.insert()

    # Check whether X is an AnnotationMatrix or not
    anno_matrix = isinstance(X, csr_AnnotationMatrix)
    if not anno_matrix:
        X = list(X)

    # Prepare values
    insert_vals = []
    for i, k, p in marginal_tuples:
        cid = X.get_candidate(session, i).id if anno_matrix else X[i].id
        insert_vals.append({
            'candidate_id': cid,
            'training': training,
            'value': k,
            # We cast p in case its a numpy type, which psycopg2 does not handle
            'probability': float(p)
        })

    # Execute update
    session.execute(q, insert_vals)
    session.commit()
    logger.info("Saved {%d} marginals".format(len(marginals)))
