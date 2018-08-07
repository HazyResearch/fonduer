import re
from itertools import chain

from fonduer.supervision.lf_helpers import (
    get_aligned_ngrams,
    get_left_ngrams,
    get_row_ngrams,
    overlap,
)


def LF_storage_row(c):
    return 1 if "storage" in get_row_ngrams(c.temp.span) else 0


def LF_temperature_row(c):
    return 1 if "temperature" in get_row_ngrams(c.temp.span) else 0


def LF_operating_row(c):
    return 1 if "operating" in get_row_ngrams(c.temp.span) else 0


def LF_tstg_row(c):
    return 1 if overlap(["tstg", "stg", "ts"], list(get_row_ngrams(c.temp.span))) else 0


def LF_to_left(c):
    return 1 if "to" in get_left_ngrams(c.temp.span, window=2) else 0


def LF_negative_number_left(c):
    return (
        1
        if any(
            [
                re.match(r"-\s*\d+", ngram)
                for ngram in get_left_ngrams(c.temp.span, window=4)
            ]
        )
        else 0
    )


def LF_test_condition_aligned(c):
    return (
        -1
        if overlap(["test", "condition"], list(get_aligned_ngrams(c.temp.span)))
        else 0
    )


def LF_collector_aligned(c):
    return (
        -1
        if overlap(
            ["collector", "collector-current", "collector-base", "collector-emitter"],
            list(get_aligned_ngrams(c.temp.span)),
        )
        else 0
    )


def LF_current_aligned(c):
    return (
        -1
        if overlap(["current", "dc", "ic"], list(get_aligned_ngrams(c.temp.span)))
        else 0
    )


def LF_voltage_row_temp(c):
    return (
        -1
        if overlap(
            ["voltage", "cbo", "ceo", "ebo", "v"], list(get_aligned_ngrams(c.temp.span))
        )
        else 0
    )


def LF_voltage_row_part(c):
    return (
        -1
        if overlap(
            ["voltage", "cbo", "ceo", "ebo", "v"], list(get_aligned_ngrams(c.temp.span))
        )
        else 0
    )


def LF_typ_row(c):
    return -1 if overlap(["typ", "typ."], list(get_row_ngrams(c.temp.span))) else 0


def LF_complement_left_row(c):
    return (
        -1
        if (
            overlap(
                ["complement", "complementary"],
                chain.from_iterable(
                    [
                        get_row_ngrams(c.part.span),
                        get_left_ngrams(c.part.span, window=10),
                    ]
                ),
            )
        )
        else 0
    )


def LF_too_many_numbers_row(c):
    num_numbers = list(get_row_ngrams(c.temp.span, attrib="ner_tags")).count("number")
    return -1 if num_numbers >= 3 else 0


def LF_temp_on_high_page_num(c):
    return -1 if c.temp.span.get_attrib_tokens("page")[0] > 2 else 0


def LF_temp_outside_table(c):
    return -1 if not c.temp.span.sentence.is_tabular() is None else 0


def LF_not_temp_relevant(c):
    return (
        -1
        if not overlap(
            ["storage", "temperature", "tstg", "stg", "ts"],
            list(get_aligned_ngrams(c.temp.span)),
        )
        else 0
    )
