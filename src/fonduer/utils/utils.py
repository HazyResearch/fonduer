import re
from builtins import range
from typing import Dict, Iterator, List, Union

from fonduer.parser.models import Sentence


def camel_to_under(name: str) -> str:
    """
    Converts camel-case string to lowercase string separated by underscores.

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
    f = (lambda x: x.lower()) if lower else (lambda x: x)
    N = len(tokens)
    for root in range(N):
        for n in range(max(n_min - 1, 0), min(n_max, N - root)):
            yield f(delim.join(tokens[root : root + n + 1]))
