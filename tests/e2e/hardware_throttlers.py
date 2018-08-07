import re

from hardware_spaces import expand_part_range

from fonduer.supervision.lf_helpers import (
    get_horz_ngrams,
    get_vert_ngrams,
    is_horz_aligned,
    is_vert_aligned,
    same_table,
)


def temp_throttler(c):
    (part, attr) = c
    if same_table((part, attr)):
        return is_horz_aligned((part, attr)) or is_vert_aligned((part, attr))
    return True


def filter_non_parts(c):
    ret = set()
    for _ in c:
        for __ in expand_part_range(_):
            if re.match("^([0-9]+[A-Z]+|[A-Z]+[0-9]+)[0-9A-Z]*$", __) and len(__) > 2:
                ret.add(__)
    return ret


def LF_part_miss_match(c):
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
    (part, attr) = c
    if same_table((part, attr)):
        return is_horz_aligned((part, attr)) or is_vert_aligned((part, attr))
    if LF_part_miss_match((part, attr)) < 0:
        return False
    return True
