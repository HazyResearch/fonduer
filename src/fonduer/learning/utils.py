import logging

import numpy as np
import torch
import torch.nn.functional as F

from fonduer.learning.models.marginal import Marginal

logger = logging.getLogger(__name__)

# ###########################################################
# # General Learning Utilities
# ###########################################################


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
    logger.info("Saved {%d} marginals".format(len(marginals)))


def reshape_marginals(marginals):
    """Returns correctly shaped marginals as np array

    :param marginals: Marginal matrix.
    :return: The reshaped marginal matrix.
    """
    # Make sure training marginals are a numpy array first
    try:
        shape = marginals.shape
    except Exception:
        marginals = np.array(marginals)
        shape = marginals.shape

    # Set cardinality + marginals in proper format for binary v. categorical
    if len(shape) != 1:
        # If k = 2, make sure is M-dim array
        if shape[1] == 2:
            marginals = marginals[:, 1].reshape(-1)
    return marginals


class LabelBalancer(object):
    """Utility class to rebalance training labels.

    :param y: Labels to balance
    :type y: numpy.array

    :Example:
        To get the indices of a training set with labels y and
        around 90 percent negative examples::

            LabelBalancer(y).get_train_idxs(rebalance=0.1)
    """

    def __init__(self, y):
        self.y = np.ravel(y)

    def _get_pos(self, split):
        return np.where(self.y > (split + 1e-6))[0]

    def _get_neg(self, split):
        return np.where(self.y < (split - 1e-6))[0]

    def _try_frac(self, m, n, pn):
        # Return (a, b) s.t. a <= m, b <= n
        # and b / a is as close to pn as possible
        r = int(round(float(pn * m) / (1.0 - pn)))
        s = int(round(float((1.0 - pn) * n) / pn))
        return (m, r) if r <= n else ((s, n) if s <= m else (m, n))

    def _get_counts(self, nneg, npos, frac_pos):
        if frac_pos > 0.5:
            return self._try_frac(nneg, npos, frac_pos)
        else:
            return self._try_frac(npos, nneg, 1.0 - frac_pos)[::-1]

    def get_train_idxs(self, rebalance=False, split=0.5, rand_state=None):
        """Get training indices based on y.

        :param rebalance: bool or fraction of positive examples desired If
            True, default fraction is 0.5. If False no balancing.
        :param split: Split point for positive and negative classes.
        :param rand_state: numpy random state to random select data indices.
        """
        rs = np.random if rand_state is None else rand_state
        pos, neg = self._get_pos(split), self._get_neg(split)
        if rebalance:
            if len(pos) == 0:
                raise ValueError("No positive labels.")
            if len(neg) == 0:
                raise ValueError("No negative labels.")
            p = 0.5 if rebalance else rebalance
            n_neg, n_pos = self._get_counts(len(neg), len(pos), p)
            pos = rs.choice(pos, size=n_pos, replace=False)
            neg = rs.choice(neg, size=n_neg, replace=False)
        idxs = np.concatenate([pos, neg])
        rs.shuffle(idxs)
        return idxs


# ##########################################################
# # Loss functions
# ##########################################################


def SoftCrossEntropyLoss(input, target):
    """
    Calculate the CrossEntropyLoss with soft targets

    :param input: prediction logits
    :param target: target probabilities
    :return: loss
    """
    total_loss = torch.tensor(0.0)
    for i in range(input.size(1)):
        cls_idx = torch.full((input.size(0),), i, dtype=torch.long)
        loss = F.cross_entropy(input, cls_idx, reduce=False)
        total_loss += target[:, i].dot(loss)
    return total_loss / input.shape[0]
