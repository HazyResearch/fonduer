"""Fonduer data model utils."""
import logging
from functools import lru_cache
from typing import Callable, Iterable, List, Set, Union

from fonduer.candidates.models import Candidate, Mention
from fonduer.candidates.models.span_mention import TemporarySpanMention


@lru_cache(maxsize=1024)
def _to_span(
    x: Union[Candidate, Mention, TemporarySpanMention], idx: int = 0
) -> TemporarySpanMention:
    """Convert a Candidate, Mention, or Span to a span."""
    if isinstance(x, Candidate):
        return x[idx].context
    elif isinstance(x, Mention):
        return x.context
    elif isinstance(x, TemporarySpanMention):
        return x
    else:
        raise ValueError(f"{type(x)} is an invalid argument type")


@lru_cache(maxsize=1024)
def _to_spans(
    x: Union[Candidate, Mention, TemporarySpanMention]
) -> List[TemporarySpanMention]:
    """Convert a Candidate, Mention, or Span to a list of spans."""
    if isinstance(x, Candidate):
        return [_to_span(m) for m in x]
    elif isinstance(x, Mention):
        return [x.context]
    elif isinstance(x, TemporarySpanMention):
        return [x]
    else:
        raise ValueError(f"{type(x)} is an invalid argument type")


def is_superset(a: Iterable, b: Iterable) -> bool:
    """Check if a is a superset of b.

    This is typically used to check if ALL of a list of sentences is in the
    ngrams returned by an lf_helper.

    :param a: A collection of items
    :param b: A collection of items
    """
    return set(a).issuperset(b)


def overlap(a: Iterable, b: Iterable) -> bool:
    """Check if a overlaps b.

    This is typically used to check if ANY of a list of sentences is in the
    ngrams returned by an lf_helper.

    :param a: A collection of items
    :param b: A collection of items
    """
    return not set(a).isdisjoint(b)


def get_matches(
    lf: Callable, candidate_set: Set[Candidate], match_values: List[int] = [1, -1]
) -> List[Candidate]:
    """Return a list of candidates that are matched by a particular LF.

    A simple helper function to see how many matches (non-zero by default) an
    LF gets.

    :param lf: The labeling function to apply to the candidate_set
    :param candidate_set: The set of candidates to evaluate
    :param match_values: An option list of the values to consider as matched.
        [1, -1] by default.
    """
    logger = logging.getLogger(__name__)
    matches: List[Candidate] = []
    for c in candidate_set:
        label = lf(c)
        if label in match_values:
            matches.append(c)
    logger.info(f"{len(matches)} matches")
    return matches
