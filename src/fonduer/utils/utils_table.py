"""Fonduer tabular utils."""
import itertools
from builtins import range
from functools import lru_cache
from typing import List, Optional, Union

from fonduer.parser.models.sentence import Sentence
from fonduer.parser.models.table import Cell


@lru_cache(maxsize=1024)
def _min_range_diff(
    a_start: int, a_end: int, b_start: int, b_end: int, absolute: bool = True
) -> int:
    """Get the minimum range difference.

    # if absolute=True, return the absolute value of minimum magnitude difference
    # if absolute=False, return the raw value of minimum magnitude difference
    # TODO: move back to efficient implementation once it sees that
    # min_range_diff(3,3,2,3) = 0 return max(0, max(a_end - b_start, b_end -
    # a_start))

    :param a_start: The start index of the first object.
    :param a_end: The end index of the first object.
    :param b_start: The start index of the second object.
    :param b_end: The end index of the second object.
    :param absolute: Whether use absolute value, defaults to True.
    :return: The minimum range difference.
    """
    f = lambda x: (abs(x) if absolute else x)
    return min(
        [
            f(ii[0] - ii[1])
            for ii in itertools.product(
                list(range(a_start, a_end + 1)), list(range(b_start, b_end + 1))
            )
        ],
        key=abs,
    )


def min_row_diff(
    a: Union[Cell, Sentence], b: Union[Cell, Sentence], absolute: bool = True
) -> int:
    """Get the minimum row difference of two sentences or cells.

    :param a: The first cell or sentence.
    :param b: The second cell or sentence.
    :param absolute: Whether use absolute value, defaults to True.
    :return: The minimum row difference.
    """
    return _min_range_diff(
        a.row_start, a.row_end, b.row_start, b.row_end, absolute=absolute
    )


def min_col_diff(
    a: Union[Cell, Sentence], b: Union[Sentence, Cell], absolute: bool = True
) -> int:
    """Get the minimum column difference of two sentences or cells.

    :param a: The first cell or sentence.
    :param b: The second cell or sentence.
    :param absolute: Whether use absolute value, defaults to True.
    :return: The minimum column difference.
    """
    return _min_range_diff(
        a.col_start, a.col_end, b.col_start, b.col_end, absolute=absolute
    )


def min_axis_diff(
    a: Union[Cell, Sentence],
    b: Union[Cell, Sentence],
    axis: Optional[str] = None,
    absolute: bool = True,
) -> int:
    """Get the minimum axis difference of two sentences or cells.

    :param a: The first cell or sentence.
    :param b: The second cell or sentence.
    :param axis: The axis to calculate the difference, defaults to None.
    :return: The minimum axis difference.
    """
    if axis == "row":
        return min_row_diff(a, b, absolute)
    elif axis == "col":
        return min_col_diff(a, b, absolute)
    else:
        return min(min_row_diff(a, b, absolute), min_col_diff(a, b, absolute))


def is_row_aligned(
    a: Union[Cell, Sentence], b: Union[Cell, Sentence], spread: List[int] = [0, 0]
) -> bool:
    """Check two sentences or cells are row-wise aligned.

    :param a: The first cell or sentence.
    :param b: The second cell or sentence.
    :param spread: Row difference range, defaults to [0, 0].
    :return: Return True if two sentences or cells are row-wise aligned.
    """
    return min_row_diff(a, b) in range(spread[0], spread[1] + 1)


def is_col_aligned(
    a: Union[Sentence, Cell], b: Union[Cell, Sentence], spread: List[int] = [0, 0]
) -> bool:
    """Check two sentences or cells are column-wise aligned.

    :param a: The first cell or sentence.
    :param b: The second cell or sentence.
    :param spread: Column difference range, defaults to [0, 0].
    :return: Return True if two sentences or cells are column-wise aligned.
    """
    return min_col_diff(a, b) in range(spread[0], spread[1] + 1)


def is_axis_aligned(
    a: Union[Cell, Sentence],
    b: Union[Cell, Sentence],
    axis: Optional[str] = None,
    spread: List[int] = [0, 0],
) -> bool:
    """Check two sentences or cells are axis-wise aligned.

    :param a: The first cell or sentence.
    :param b: The second cell or sentence.
    :param axis: The axis to calculate the alignment, defaults to None.
    :param spread: Row/column difference range, defaults to [0, 0].
    :return: Return True if two sentences or cells are axis-wise aligned.
    """
    if axis == "row":
        return is_row_aligned(a, b, spread=spread)
    elif axis == "col":
        return is_col_aligned(a, b, spread=spread)
    else:
        return is_row_aligned(a, b, spread=spread) or is_col_aligned(
            a, b, spread=spread
        )


def num_rows(a: Union[Cell, Sentence]) -> int:
    """Get number of rows that sentence or cell spans.

    :param a: The cell or sentence.
    :return: The number of rows that sentence or cell spans.
    """
    return a.row_start - a.row_end + 1


def num_cols(a: Union[Cell, Sentence]) -> int:
    """Get number of columns that sentence or cell spans.

    :param a: The cell or sentence.
    :return: The number of columns that sentence or cell spans.
    """
    return a.col_start - a.col_end + 1
