import logging


def is_superset(a, b):
    """Check if a is a superset of b.

    This is typically used to check if ALL of a list of sentences is in the
    ngrams returned by an lf_helper.

    :param a: A collection of items
    :param b: A collection of items
    :rtype: boolean
    """
    return set(a).issuperset(b)


def overlap(a, b):
    """Check if a overlaps b.

    This is typically used to check if ANY of a list of sentences is in the
    ngrams returned by an lf_helper.

    :param a: A collection of items
    :param b: A collection of items
    :rtype: boolean
    """
    return not set(a).isdisjoint(b)


def get_matches(lf, candidate_set, match_values=[1, -1]):
    """Return a list of candidates that are matched by a particular LF.

    A simple helper function to see how many matches (non-zero by default) an
    LF gets.

    :param lf: The labeling function to apply to the candidate_set
    :param candidate_set: The set of candidates to evaluate
    :param match_values: An option list of the values to consider as matched.
        [1, -1] by default.
    :rtype: a list of candidates
    """
    logger = logging.getLogger(__name__)
    matches = []
    for c in candidate_set:
        label = lf(c)
        if label in match_values:
            matches.append(c)
    logger.info(("%s matches") % len(matches))
    return matches
