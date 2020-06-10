"""Hardware throttlers."""
import re

from fonduer.utils.data_model_utils import (
    get_horz_ngrams,
    get_vert_ngrams,
    is_horz_aligned,
    is_vert_aligned,
    same_table,
)
from tests.shared.hardware_spaces import expand_part_range


def temp_throttler(c):
    """Temperature throttler."""
    (part, attr) = c
    if same_table((part, attr)):
        return is_horz_aligned((part, attr)) or is_vert_aligned((part, attr))
    return True


def filter_non_parts(c):
    """Filter non parts."""
    ret = set()
    for _ in c:
        for __ in expand_part_range(_):
            if re.match("^([0-9]+[A-Z]+|[A-Z]+[0-9]+)[0-9A-Z]*$", __) and len(__) > 2:
                ret.add(__)
    return ret


def LF_part_miss_match(c):
    """Return 0 if part mismatch."""
    ngrams_part = set(list(get_vert_ngrams(c[1], n_max=1)))
    ngrams_part = filter_non_parts(
        ngrams_part.union(set(list(get_horz_ngrams(c[1], n_max=1))))
    )
    return (
        0
        if len(ngrams_part) == 0
        or any([c[0].get_span().lower().startswith(_.lower()) for _ in ngrams_part])
        else -1
    )


def volt_throttler(c):
    """Voltage throttler."""
    (part, attr) = c
    if same_table((part, attr)):
        return is_horz_aligned((part, attr)) or is_vert_aligned((part, attr))
    if LF_part_miss_match((part, attr)) < 0:
        return False
    return True
