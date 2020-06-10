"""Hardware labeling functions."""
import re
from itertools import chain

from fonduer.utils.data_model_utils import (
    get_aligned_ngrams,
    get_left_ngrams,
    get_row_ngrams,
    overlap,
)
from tests.shared.hardware_utils import ABSTAIN, FALSE, TRUE


def LF_storage_row(c):
    """Return True if temp mention's row ngrams contain ``storage''."""
    return TRUE if "storage" in get_row_ngrams(c.temp) else ABSTAIN


def LF_temperature_row(c):
    """Return True if temp mention's row ngrams contain ``temperature''."""
    return TRUE if "temperature" in get_row_ngrams(c.temp) else ABSTAIN


def LF_operating_row(c):
    """Return True if temp mention's row ngrams contain ``operating''."""
    return TRUE if "operating" in get_row_ngrams(c.temp) else ABSTAIN


def LF_tstg_row(c):
    """Return True if temp mention's row ngrams overlap with the following keywords."""
    return (
        TRUE
        if overlap(["tstg", "stg", "ts"], list(get_row_ngrams(c.temp)))
        else ABSTAIN
    )


def LF_to_left(c):
    """Return True if temp mention's left ngrams contain ``to''."""
    return TRUE if "to" in get_left_ngrams(c.temp, window=2) else ABSTAIN


def LF_negative_number_left(c):
    """Return True if temp mention's left ngrams contain negative number."""
    return (
        TRUE
        if any(
            [re.match(r"-\s*\d+", ngram) for ngram in get_left_ngrams(c.temp, window=4)]
        )
        else ABSTAIN
    )


def LF_test_condition_aligned(c):
    """Return False if temp mention's ngrams align with ``test'' or ``condition''."""
    return (
        FALSE
        if overlap(["test", "condition"], list(get_aligned_ngrams(c.temp)))
        else ABSTAIN
    )


def LF_collector_aligned(c):
    """Return False if temp mention's ngrams align with the following keywords."""
    return (
        FALSE
        if overlap(
            ["collector", "collector-current", "collector-base", "collector-emitter"],
            list(get_aligned_ngrams(c.temp)),
        )
        else ABSTAIN
    )


def LF_current_aligned(c):
    """Return False if temp mention's ngrams align with the following keywords."""
    return (
        FALSE
        if overlap(["current", "dc", "ic"], list(get_aligned_ngrams(c.temp)))
        else ABSTAIN
    )


def LF_voltage_row_temp(c):
    """Return False if temp mention's ngrams align with the following keywords."""
    return (
        FALSE
        if overlap(
            ["voltage", "cbo", "ceo", "ebo", "v"], list(get_aligned_ngrams(c.temp))
        )
        else ABSTAIN
    )


def LF_voltage_row_part(c):
    """Return False if temp mention's ngrams align with the following keywords."""
    return (
        FALSE
        if overlap(
            ["voltage", "cbo", "ceo", "ebo", "v"], list(get_aligned_ngrams(c.temp))
        )
        else ABSTAIN
    )


def LF_typ_row(c):
    """Return False if temp mention's ngrams align with the following keywords."""
    return FALSE if overlap(["typ", "typ."], list(get_row_ngrams(c.temp))) else ABSTAIN


def LF_complement_left_row(c):
    """Return False if temp mention's ngrams align with the following keywords."""
    return (
        FALSE
        if (
            overlap(
                ["complement", "complementary"],
                chain.from_iterable(
                    [get_row_ngrams(c.part), get_left_ngrams(c.part, window=10)]
                ),
            )
        )
        else ABSTAIN
    )


def LF_too_many_numbers_row(c):
    """Return False if too many numbers in the row."""
    num_numbers = list(get_row_ngrams(c.temp, attrib="ner_tags")).count("number")
    return FALSE if num_numbers >= 3 else ABSTAIN


def LF_temp_on_high_page_num(c):
    """Return False if temp mention on high page number."""
    return FALSE if c.temp.context.get_attrib_tokens("page")[0] > 2 else ABSTAIN


def LF_temp_outside_table(c):
    """Return False if temp mention is outside of the table."""
    return FALSE if not c.temp.context.sentence.is_tabular() is None else ABSTAIN


def LF_not_temp_relevant(c):
    """Return False if temp mention's ngrams overlap with the following keywords."""
    return (
        FALSE
        if not overlap(
            ["storage", "temperature", "tstg", "stg", "ts"],
            list(get_aligned_ngrams(c.temp)),
        )
        else ABSTAIN
    )


# Voltage LFS


def LF_bad_keywords_in_row(c):
    """Return False if volt mention's ngrams overlap with the following keywords."""
    return (
        FALSE
        if overlap(
            ["continuous", "cut-off", "gain", "breakdown"], get_row_ngrams(c.volt)
        )
        else ABSTAIN
    )


def LF_current_in_row(c):
    """Return False if volt mention's ngrams overlap with the following keywords."""
    return FALSE if overlap(["i", "ic", "mA"], get_row_ngrams(c.volt)) else ABSTAIN


non_ce_voltage_keywords = set(
    [
        "collector-base",
        "collector - base",
        "collector base",
        "vcbo",
        "cbo",
        "vces",
        "emitter-base",
        "emitter - base",
        "emitter base",
        "vebo",
        "ebo",
        "breakdown voltage",
        "emitter breakdown",
        "emitter breakdown voltage",
        "current",
    ]
)


def LF_non_ce_voltages_in_row(c):
    """Return False if volt mention's ngrams overlap with non_ce_voltage_keywords."""
    return (
        FALSE
        if overlap(non_ce_voltage_keywords, get_row_ngrams(c.volt, n_max=3))
        else ABSTAIN
    )
