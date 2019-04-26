#! /usr/bin/env python
import logging

from fonduer.candidates.models import Candidate
from fonduer.learning.utils import entity_confusion_matrix


def test_entity_confusion_matrix(caplog):
    """Test the confusion matrix."""
    caplog.set_level(logging.INFO)

    # Synthesize candidates
    cand1 = Candidate(id=1, type="type", split=0)
    cand2 = Candidate(id=2, type="type", split=0)
    cand3 = Candidate(id=3, type="type", split=0)
    cand4 = Candidate(id=4, type="type", split=0)

    # pred and gold as set
    pred = {cand1, cand2, cand3}
    gold = {cand1, cand2, cand4}
    (TP, FP, FN) = entity_confusion_matrix(pred, gold)

    assert TP == {cand1, cand2}
    assert FP == {cand3}
    assert FN == {cand4}

    # pred as list
    pred = [cand1, cand2, cand3]
    (TP, FP, FN) = entity_confusion_matrix(pred, gold)

    assert TP == {cand1, cand2}
    assert FP == {cand3}
    assert FN == {cand4}

    # test if the order of elements does not affect the output
    pred = [cand3, cand2, cand1]
    (TP, FP, FN) = entity_confusion_matrix(pred, gold)

    assert TP == {cand1, cand2}
    assert FP == {cand3}
    assert FN == {cand4}
