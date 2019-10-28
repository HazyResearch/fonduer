from fonduer.candidates.models import Candidate
from fonduer.learning.utils import confusion_matrix


def test_confusion_matrix():
    """Test the confusion matrix."""
    # Synthesize candidates
    cand1 = Candidate(id=1, type="type")
    cand2 = Candidate(id=2, type="type")
    cand3 = Candidate(id=3, type="type")
    cand4 = Candidate(id=4, type="type")

    # pred and gold as set
    pred = {cand1, cand2, cand3}
    gold = {cand1, cand2, cand4}
    (TP, FP, FN) = confusion_matrix(pred, gold)

    assert TP == {cand1, cand2}
    assert FP == {cand3}
    assert FN == {cand4}

    # pred as list
    pred = [cand1, cand2, cand3]
    (TP, FP, FN) = confusion_matrix(pred, gold)

    assert TP == {cand1, cand2}
    assert FP == {cand3}
    assert FN == {cand4}

    # test if the order of elements does not affect the output
    pred = [cand3, cand2, cand1]
    (TP, FP, FN) = confusion_matrix(pred, gold)

    assert TP == {cand1, cand2}
    assert FP == {cand3}
    assert FN == {cand4}

    # Assume the followings are entities
    pred = {"1", "2", "3"}
    gold = {"1", "2", "4"}
    (TP, FP, FN) = confusion_matrix(pred, gold)

    assert TP == {"1", "2"}
    assert FP == {"3"}
    assert FN == {"4"}
