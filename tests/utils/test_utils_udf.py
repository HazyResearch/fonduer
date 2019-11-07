import logging

import numpy as np

from fonduer.utils.utils_udf import shift_label_matrix, unshift_label_matrix


def test_shift_label_matrix(caplog):
    """Test the label matrix shifter and unshifter."""
    caplog.set_level(logging.INFO)

    """
    L is a dense label matrix (ABSTAIN as -1) with values:
    -1  0
     1 -1
    """
    L = np.array([[-1, 0], [1, -1]])
    """
    L_sparse is a sparse label matrix (ABSTAIN as 0)
     0  1
     2  0
    """
    L_sparse = shift_label_matrix(L)
    assert np.array_equal(L, unshift_label_matrix(L_sparse))
    assert L_sparse.count_nonzero() == 2
