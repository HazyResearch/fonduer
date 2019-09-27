import tempfile

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fonduer import init_logging
from fonduer.candidates.candidates import Candidate
from fonduer.learning.classifier import Classifier
from fonduer.learning.disc_models.logistic_regression import LogisticRegression
from tests.shared.hardware_lfs import FALSE, TRUE


def test_classifier_predict(caplog):
    """Test Classifier#predict."""

    init_logging(log_dir=tempfile.gettempdir())
    # Create dummy candidates and their features
    C = [
        Candidate(id=1, type="type"),
        Candidate(id=2, type="type"),
        Candidate(id=3, type="type"),
    ]
    F = csr_matrix((len(C), 1), dtype=np.int8)
    classifier = Classifier()

    # binary
    classifier.cardinality = 2

    # mock the marginals so that 0.4 for FALSE and 0.6 for TRUE
    def mock_marginals(X):
        marginal = np.ones((len(X), 1)) * 0.4
        return np.concatenate((marginal, 1 - marginal), axis=1)

    classifier.marginals = mock_marginals

    # Classifier#predict takes an array of candidates: C.
    Y_pred = classifier.predict(C, b=0.6, pos_label=TRUE)
    np.testing.assert_array_equal(Y_pred, np.array([FALSE, FALSE, FALSE]))

    # Classifier#predict takes an array of candidates (not a tuple of C and F).
    # TODO: Classifier#_preprocess_data should assume that
    #       X to be a tuple of C and F like its subclasses?
    Y_pred = classifier.predict((C, F), b=0.6, pos_label=TRUE)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(Y_pred, np.array([FALSE, FALSE, FALSE]))


def test_logistic_regression_predict(caplog):
    """Test LogisticRegression#predict."""

    init_logging(log_dir=tempfile.gettempdir())
    # Create dummy candidates and their features
    C = [
        Candidate(id=1, type="type"),
        Candidate(id=2, type="type"),
        Candidate(id=3, type="type"),
    ]
    F = csr_matrix((len(C), 1), dtype=np.int8)
    classifier = LogisticRegression()

    # binary
    classifier.cardinality = 2

    # mock the marginals so that 0.4 for FALSE and 0.6 for TRUE
    def mock_marginals(X):
        marginal = np.ones((len(X), 1)) * 0.4
        return np.concatenate((marginal, 1 - marginal), axis=1)

    classifier.marginals = mock_marginals

    # marginal < b gives FALSE predictions.
    Y_pred = classifier.predict((C, F), b=0.6, pos_label=TRUE)
    np.testing.assert_array_equal(Y_pred, np.array([FALSE, FALSE, FALSE]))

    # b <= marginal gives TRUE predictions.
    Y_pred = classifier.predict((C, F), b=0.59, pos_label=TRUE)
    np.testing.assert_array_equal(Y_pred, np.array([TRUE, TRUE, TRUE]))

    # return_probs=True should return marginals too.
    _, Y_prob = classifier.predict((C, F), b=0.59, pos_label=TRUE, return_probs=True)
    np.testing.assert_array_equal(Y_prob, mock_marginals(C))

    # When cardinality == 2, pos_label other than [1, 2] raises an error
    with pytest.raises(ValueError):
        Y_pred = classifier.predict((C, F), b=0.6, pos_label=3)

    # tertiary
    classifier.cardinality = 3

    # mock the marginals so that 0.2 for class 1, 0.2 for class 2, and 0.6 for class 3
    def mock_marginals(X):
        a = np.ones((len(X), 1)) * 0.2
        b = np.ones((len(X), 1)) * 0.2
        return np.concatenate((a, b, 1 - a - b), axis=1)

    classifier.marginals = mock_marginals

    # class 3 has the largest marginal.
    Y_pred = classifier.predict((C, F))
    np.testing.assert_array_equal(Y_pred, np.array([3, 3, 3]))

    # class 3 has 0.6 of marginal.
    _, Y_prob = classifier.predict((C, F), return_probs=True)
    np.testing.assert_array_equal(Y_prob, mock_marginals(C))
