import logging

import numpy as np
import scipy.sparse as sparse
from pandas import DataFrame, Series

from fonduer.candidates.models import Candidate
from fonduer.supervision.models import GoldLabel, GoldLabelKey
from fonduer.utils.annotator import Annotator
from fonduer.utils.utils import (
    matrix_conflicts,
    matrix_coverage,
    matrix_fn,
    matrix_fp,
    matrix_overlaps,
    matrix_tn,
    matrix_tp,
)
from fonduer.utils.utils_annotations import csr_AnnotationMatrix

logger = logging.getLogger(__name__)


class LabelAnnotator(Annotator):
    def __init__(self, candidate_type, lfs, label_generator=None, **kwargs):

        if lfs is not None:
            labels = lambda c: [(lf.__name__, lf(c)) for lf in lfs]
        elif label_generator is not None:
            labels = lambda c: label_generator(c)
        else:
            raise ValueError("Must provide lfs or label_generator kwarg.")

        # Convert lfs to a generator function
        # In particular, catch verbose values and convert to integer ones
        def f_gen(c):
            for lf_key, label in labels(c):
                # Note: We assume if the LF output is an int, it is already
                # mapped correctly
                if isinstance(label, int):
                    yield lf_key, label
                # None is a protected LF output value corresponding to 0,
                # representing LF abstaining
                elif label is None:
                    yield lf_key, 0
                elif label in c.values:
                    if c.cardinality > 2:
                        yield lf_key, c.values.index(label) + 1
                    # Note: Would be nice to not special-case here, but for
                    # consistency we leave binary LF range as {-1,0,1}
                    else:
                        val = 1 if c.values.index(label) == 0 else -1
                        yield lf_key, val
                else:
                    raise ValueError(
                        "Can't parse label value {} for candidate values {}".format(
                            label, c.values
                        )
                    )

        super(LabelAnnotator, self).__init__(
            candidate_type, annotation_type="label", f=f_gen, **kwargs
        )


def _load_matrix(
    matrix_class,
    annotation_key_class,
    annotation_class,
    session,
    split=0,
    cids_query=None,
    key_group=0,
    key_names=None,
    zero_one=False,
    load_as_array=False,
):
    """
    Returns the annotations corresponding to a split of candidates with N members
    and an AnnotationKey group with M distinct keys as an N x M CSR sparse matrix.
    """
    cid_query = cids_query or session.query(Candidate.id).filter(
        Candidate.split == split
    )
    cid_query = cid_query.order_by(Candidate.id)

    keys_query = session.query(annotation_key_class.id)
    keys_query = keys_query.filter(annotation_key_class.group == key_group)
    if key_names is not None:
        keys_query = keys_query.filter(
            annotation_key_class.name.in_(frozenset(key_names))
        )
    keys_query = keys_query.order_by(annotation_key_class.id)

    # First, we query to construct the row index map
    cid_to_row = {}
    row_to_cid = {}
    for (cid,) in cid_query.all():
        if cid not in cid_to_row:
            j = len(cid_to_row)

            # Create both mappings
            cid_to_row[cid] = j
            row_to_cid[j] = cid

    # Second, we query to construct the column index map
    kid_to_col = {}
    col_to_kid = {}
    for (kid,) in keys_query.all():
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
        (data, (row, columns)), shape=(len(cid_to_row), len(kid_to_col))
    )

    # Return as an AnnotationMatrix
    Xr = matrix_class(
        X, candidate_index=cid_to_row, row_index=row_to_cid, key_index=kid_to_col
    )
    return np.squeeze(Xr.toarray()) if load_as_array else Xr


class csr_LabelMatrix(csr_AnnotationMatrix):
    def lf_stats(self, session, labels=None, est_accs=None):
        """Returns a pandas DataFrame with the LFs and various per-LF statistics"""
        lf_names = [self.get_key(session, j).name for j in range(self.shape[1])]

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
            d["Learned Acc."] = est_accs
            d["Learned Acc."].index = lf_names
        return DataFrame(data=d, index=lf_names)[col_names]


def load_gold_labels(session, annotator_name, **kwargs):
    return _load_matrix(
        csr_LabelMatrix,
        GoldLabelKey,
        GoldLabel,
        session,
        key_names=[annotator_name],
        **kwargs
    )
