"""Fonduer utils."""
import re
from builtins import range
from typing import TYPE_CHECKING, Dict, Iterator, List, Set, Tuple, Type, Union

from fonduer.parser.models import Context, Document, Sentence

if TYPE_CHECKING:  # to prevent circular imports
    from fonduer.candidates.models import Candidate


def camel_to_under(name: str) -> str:
    """
    Convert camel-case string to lowercase string separated by underscores.

    Written by epost (http://stackoverflow.com/questions/1175208).

    :param name: String to be converted
    :return: new String with camel-case converted to lowercase, underscored
    """
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def get_as_dict(x: Union[Dict, Sentence]) -> Dict:
    """Return an object as a dictionary of its attributes."""
    if isinstance(x, dict):
        return x
    else:
        try:
            return x._asdict()
        except AttributeError:
            return x.__dict__


def tokens_to_ngrams(
    tokens: List[str],
    n_min: int = 1,
    n_max: int = 3,
    delim: str = " ",
    lower: bool = False,
) -> Iterator[str]:
    """Get n-grams from tokens."""
    f = (lambda x: x.lower()) if lower else (lambda x: x)
    N = len(tokens)
    for root in range(N):
        for n in range(max(n_min - 1, 0), min(n_max, N - root)):
            yield f(delim.join(tokens[root : root + n + 1]))


def get_set_of_stable_ids(
    doc: Document, candidate_class: "Type[Candidate]"
) -> Set[Tuple[str, ...]]:
    """Return a set of stable_ids of candidates.

    A stable_id of a candidate is a tuple of stable_id of the constituent context.
    """
    set_of_stable_ids = set()
    # "s" is required due to the relationship between Document and candidate_class.
    if hasattr(doc, candidate_class.__tablename__ + "s"):
        set_of_stable_ids.update(
            set(
                [
                    tuple(m.context.get_stable_id() for m in c) if c else None
                    for c in getattr(doc, candidate_class.__tablename__ + "s")
                ]
            )
        )
    return set_of_stable_ids


def get_dict_of_stable_id(doc: Document) -> Dict[str, Context]:
    """Return a mapping of a stable_id to its context."""
    return {
        doc.stable_id: doc,
        **{
            c.stable_id: c
            for a in [
                "sentences",
                "paragraphs",
                "captions",
                "cells",
                "tables",
                "sections",
                "figures",
            ]
            for c in getattr(doc, a)
        },
        **{
            c.stable_id: c
            for s in doc.sentences
            for a in ["spans", "implicit_spans"]
            for c in getattr(s, a)
        },
    }
