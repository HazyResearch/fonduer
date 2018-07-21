import logging

from fonduer.supervision.utils.structural import *
from fonduer.supervision.utils.tabular import *
from fonduer.supervision.utils.textual import *
from fonduer.supervision.utils.visual import *


def get_matches(lf, candidate_set, match_values=[1, -1]):
    """Return a list of candidates that are matched by a particular LF.

    A simple helper function to see how many matches (non-zero by default) an LF gets.

    :param lf: The labeling function to apply to the candidate_set
    :param candidate_set: The set of candidates to evaluate
    :param match_values: An option list of the values to consider as matched. [1, -1] by default.
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
