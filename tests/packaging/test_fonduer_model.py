from typing import Any, Dict, List

import numpy as np

from fonduer.packaging import F_matrix, L_matrix


def test_F_matrix():
    """Test F_matrix."""
    features: List[Dict[str, Any]] = [
        {"keys": ["key1", "key2"], "values": [0.0, 0.1]},
        {"keys": ["key1", "key2"], "values": [1.0, 1.1]},
    ]
    key_names: List[str] = ["key1", "key2"]

    F = F_matrix(features, key_names)
    D = np.array([[0.0, 0.1], [1.0, 1.1]])
    assert (F.todense() == D).all()


def test_F_matrix_limited_keys():
    """Test F_matrix with limited keys."""
    features: List[Dict[str, Any]] = [
        {"keys": ["key1", "key2"], "values": [0.0, 0.1]},
        {"keys": ["key1", "key2"], "values": [1.0, 1.1]},
    ]

    F = F_matrix(features, ["key1"])
    D = np.array([[0.0], [1.0]])
    assert (F.todense() == D).all()


def test_L_matrix():
    """Test L_matrix."""
    labels: List[Dict[str, Any]] = [
        {"keys": ["key1", "key2"], "values": [0, 1]},
        {"keys": ["key1", "key2"], "values": [1, 2]},
    ]
    key_names: List[str] = ["key1", "key2"]

    L = L_matrix(labels, key_names)
    D = np.array([[-1, 0], [0, 1]])
    assert (L == D).all()
