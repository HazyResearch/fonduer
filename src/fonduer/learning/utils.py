import logging
from typing import Set, Tuple

import numpy as np
from torch.utils.data import Dataset

from fonduer.learning.models.marginal import Marginal

logger = logging.getLogger(__name__)


def save_marginals(session, X, marginals, training=True):
    """Save marginal probabilities for a set of Candidates to db.

    :param X: A list of arbitrary objects with candidate ids accessible via a
        .id attrib
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
    except Exception:
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
    session.query(Marginal).filter(Marginal.training == training).delete(
        synchronize_session="fetch"
    )

    # Prepare bulk INSERT query
    q = Marginal.__table__.insert()

    # Prepare values
    insert_vals = []
    for i, k, p in marginal_tuples:
        cid = X[i].id
        insert_vals.append(
            {
                "candidate_id": cid,
                "training": training,
                "value": k,
                # We cast p in case its a numpy type, which psycopg2 does not handle
                "probability": float(p),
            }
        )

    # Execute update
    session.execute(q, insert_vals)
    session.commit()
    logger.info(f"Saved {len(marginals)} marginals")


class MultiModalDataset(Dataset):
    """A dataset contains all multimodal features in X and coressponding label Y.
    """

    def __init__(self, X, Y=None):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.Y is not None:
            return self.X[idx], self.Y[idx]
        return self.X[idx]


def confusion_matrix(pred: Set, gold: Set) -> Tuple[Set, Set, Set]:
    """Return a confusion matrix.

    This can be used for both entity-level and mention-level

    :param pred: a set of predicted entities/candidates
    :type pred: set
    :param gold: a set of golden entities/candidates
    :type gold: set
    :return: a tuple of TP, FP, and FN
    :rtype: (set, set, set)
    """
    if not isinstance(pred, set):
        pred = set(pred)
    if not isinstance(gold, set):
        gold = set(gold)
    TP = pred.intersection(gold)
    FP = pred.difference(gold)
    FN = gold.difference(pred)
    return (TP, FP, FN)
