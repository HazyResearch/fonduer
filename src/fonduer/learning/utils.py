"""Fonduer learning utils."""
import logging
from collections import Counter
from typing import Dict, List, Set, Tuple, Union

import numpy as np
from sqlalchemy.orm import Session

from fonduer.candidates.models import Candidate, Mention
from fonduer.learning.models.marginal import Marginal

logger = logging.getLogger(__name__)


def save_marginals(
    session: Session, X: List[Candidate], marginals: Session, training: bool = True
) -> None:
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


def confusion_matrix(pred: Set, gold: Set) -> Tuple[Set, Set, Set]:
    """Return a confusion matrix.

    This can be used for both entity-level and mention-level

    :param pred: a set of predicted entities/candidates
    :param gold: a set of golden entities/candidates
    :return: a tuple of TP, FP, and FN
    """
    if not isinstance(pred, set):
        pred = set(pred)
    if not isinstance(gold, set):
        gold = set(gold)
    TP = pred.intersection(gold)
    FP = pred.difference(gold)
    FN = gold.difference(pred)
    return (TP, FP, FN)


def mention_to_tokens(
    mention: Mention, token_type: str = "words", lowercase: bool = False
) -> List[str]:
    """Extract tokens from the mention.

    :param mention: mention object.
    :param token_type: token type that wants to extract (e.g. words, lemmas, poses).
    :param lowercase: use lowercase or not.
    :return: The token list.
    """
    tokens = getattr(mention.context.sentence, token_type)
    return [w.lower() if lowercase else w for w in tokens]


def mark(l: int, h: int, idx: int) -> List[Tuple[int, str]]:
    """Produce markers based on argument positions.

    :param l: sentence position of first word in argument.
    :param h: sentence position of last word in argument.
    :param idx: argument index (1 or 2).
    :return: markers.
    """
    return [(l, f"~~[[{idx}"), (h + 1, f"{idx}]]~~")]


def mark_sentence(s: List[str], args: List[Tuple[int, int, int]]) -> List[str]:
    """Insert markers around relation arguments in word sequence.

    :param s: list of tokens in sentence.
    :param args: list of triples (l, h, idx) as per @_mark(...) corresponding
               to relation arguments
    :return: The marked sentence.

    Example:
         Then Barack married Michelle.
         ->  Then ~~[[1 Barack 1]]~~ married ~~[[2 Michelle 2]]~~.

    """
    marks = sorted([y for m in args for y in mark(*m)], reverse=True)
    x = list(s)
    for k, v in marks:
        x.insert(k, v)
    return x


def collect_word_counter(
    candidates: Union[List[Candidate], List[List[Candidate]]]
) -> Dict[str, int]:
    """Collect word counter from candidates.

    :param candidates: The candidates used to collect word counter.
    :return: The word counter.
    """
    word_counter: Counter = Counter()

    if isinstance(candidates[0], list):
        candidates = [cand for candidate in candidates for cand in candidate]
    for candidate in candidates:
        for mention in candidate:
            word_counter.update(mention_to_tokens(mention))
    return word_counter
