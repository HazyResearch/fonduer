"""Fonduer tabular modality utilities."""
from builtins import range
from collections import defaultdict
from functools import lru_cache
from itertools import chain
from typing import DefaultDict, Iterator, List, Optional, Set, Tuple, Union

import deprecation

from fonduer import __version__
from fonduer.candidates.models import Candidate, Mention
from fonduer.candidates.models.span_mention import TemporarySpanMention
from fonduer.parser.models.sentence import Sentence
from fonduer.parser.models.table import Cell, Table
from fonduer.utils.data_model_utils.textual import (
    get_neighbor_sentence_ngrams as get_neighbor_sentence_ngrams_in_textual,
    get_sentence_ngrams as get_sentence_ngrams_in_textual,
    same_sentence as same_sentence_in_textual,
)
from fonduer.utils.data_model_utils.utils import _to_span, _to_spans
from fonduer.utils.utils import tokens_to_ngrams
from fonduer.utils.utils_table import (
    is_axis_aligned,
    is_col_aligned,
    is_row_aligned,
    min_col_diff,
    min_row_diff,
)


def same_table(c: Candidate) -> bool:
    """Return True if all Mentions in the given candidate are from the same Table.

    :param c: The candidate whose Mentions are being compared
    """
    return all(
        _to_span(c[i]).sentence.is_tabular()
        and _to_span(c[i]).sentence.table == _to_span(c[0]).sentence.table
        for i in range(len(c))
    )


def same_row(c: Candidate) -> bool:
    """Return True if all Mentions in the given candidate are from the same Row.

    :param c: The candidate whose Mentions are being compared
    """
    return same_table(c) and all(
        is_row_aligned(_to_span(c[i]).sentence, _to_span(c[0]).sentence)
        for i in range(len(c))
    )


def same_col(c: Candidate) -> bool:
    """Return True if all Mentions in the given candidate are from the same Col.

    :param c: The candidate whose Mentions are being compared
    """
    return same_table(c) and all(
        is_col_aligned(_to_span(c[i]).sentence, _to_span(c[0]).sentence)
        for i in range(len(c))
    )


def is_tabular_aligned(c: Candidate) -> bool:
    """Return True if all Mentions in the given candidate are from the same Row or Col.

    :param c: The candidate whose Mentions are being compared
    """
    return same_table(c) and all(
        is_col_aligned(_to_span(c[i]).sentence, _to_span(c[0]).sentence)
        or is_row_aligned(_to_span(c[i]).sentence, _to_span(c[0]).sentence)
        for i in range(len(c))
    )


def same_cell(c: Candidate) -> bool:
    """Return True if all Mentions in the given candidate are from the same Cell.

    :param c: The candidate whose Mentions are being compared
    """
    return all(
        _to_span(c[i]).sentence.cell is not None
        and _to_span(c[i]).sentence.cell == _to_span(c[0]).sentence.cell
        for i in range(len(c))
    )


@deprecation.deprecated(
    deprecated_in="0.8.3",
    removed_in="0.9.0",
    current_version=__version__,
    details="Use :func:`textual.same_sentence()` instead",
)
def same_sentence(c: Candidate) -> bool:
    """Return True if all Mentions in the given candidate are from the same Sentence.

    :param c: The candidate whose Mentions are being compared
    """
    return same_sentence_in_textual(c)


def get_max_col_num(
    mention: Union[Candidate, Mention, TemporarySpanMention]
) -> Optional[int]:
    """Return the largest column number that a Mention occupies.

    :param mention: The Mention to evaluate. If a candidate is given, default
        to its last Mention.
    """
    span = _to_span(mention, idx=-1)
    if span.sentence.is_tabular():
        return span.sentence.cell.col_end
    else:
        return None


def get_min_col_num(
    mention: Union[Candidate, Mention, TemporarySpanMention]
) -> Optional[int]:
    """Return the lowest column number that a Mention occupies.

    :param mention: The Mention to evaluate. If a candidate is given, default
        to its first Mention.
    """
    span = _to_span(mention)
    if span.sentence.is_tabular():
        return span.sentence.cell.col_start
    else:
        return None


def get_max_row_num(
    mention: Union[Candidate, Mention, TemporarySpanMention]
) -> Optional[int]:
    """Return the largest row number that a Mention occupies.

    :param mention: The Mention to evaluate. If a candidate is given, default
        to its last Mention.
    """
    span = _to_span(mention, idx=-1)
    if span.sentence.is_tabular():
        return span.sentence.cell.row_end
    else:
        return None


def get_min_row_num(
    mention: Union[Candidate, Mention, TemporarySpanMention]
) -> Optional[int]:
    """Return the lowest row number that a Mention occupies.

    :param mention: The Mention to evaluate. If a candidate is given, default
        to its first Mention.
    """
    span = _to_span(mention)
    if span.sentence.is_tabular():
        return span.sentence.cell.row_start
    else:
        return None


@deprecation.deprecated(
    deprecated_in="0.8.3",
    removed_in="0.9.0",
    current_version=__version__,
    details="Use :func:`textual.get_sentence_ngrams()` instead",
)
def get_sentence_ngrams(
    mention: Union[Candidate, Mention, TemporarySpanMention],
    attrib: str = "words",
    n_min: int = 1,
    n_max: int = 1,
    lower: bool = True,
) -> Iterator[str]:
    """Get the ngrams that are in the Sentence of the given Mention, not including itself.

    Note that if a candidate is passed in, all of its Mentions will be
    searched.

    :param mention: The Mention whose Sentence is being searched
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    """
    return get_sentence_ngrams_in_textual(mention, attrib, n_min, n_max, lower)


@deprecation.deprecated(
    deprecated_in="0.8.3",
    removed_in="0.9.0",
    current_version=__version__,
    details="Use :func:`textual.get_neighbor_sentence_ngrams()` instead",
)
def get_neighbor_sentence_ngrams(
    mention: Union[Candidate, Mention, TemporarySpanMention],
    d: int = 1,
    attrib: str = "words",
    n_min: int = 1,
    n_max: int = 1,
    lower: bool = True,
) -> Iterator[str]:
    """Get the ngrams that are in the neighoring Sentences of the given Mention.

    Note that if a candidate is passed in, all of its Mentions will be searched.

    :param mention: The Mention whose neighbor Sentences are being searched
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    """
    return get_neighbor_sentence_ngrams_in_textual(
        mention, d, attrib, n_min, n_max, lower
    )


def get_cell_ngrams(
    mention: Union[Candidate, Mention, TemporarySpanMention],
    attrib: str = "words",
    n_min: int = 1,
    n_max: int = 1,
    lower: bool = True,
) -> Iterator[str]:
    """Get the ngrams that are in the Cell of the given mention, not including itself.

    Note that if a candidate is passed in, all of its Mentions will be searched.
    Also note that if the mention is not tabular, nothing will be yielded.

    :param mention: The Mention whose Cell is being searched
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    """
    spans = _to_spans(mention)
    for span in spans:
        if not span.sentence.is_tabular():
            continue

        for ngram in get_sentence_ngrams(
            span, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower
        ):
            yield ngram
            for ngram in chain.from_iterable(
                [
                    tokens_to_ngrams(
                        getattr(sentence, attrib), n_min=n_min, n_max=n_max, lower=lower
                    )
                    for sentence in _get_table_cells(span.sentence.table)[
                        span.sentence.cell
                    ]
                    if sentence != span.sentence
                ]
            ):
                yield ngram


def get_neighbor_cell_ngrams(
    mention: Union[Candidate, Mention, TemporarySpanMention],
    dist: int = 1,
    directions: bool = False,
    attrib: str = "words",
    n_min: int = 1,
    n_max: int = 1,
    lower: bool = True,
) -> Iterator[Union[str, Tuple[str, str]]]:
    """Get ngrams from all neighbor Cells.

    Get the ngrams from all Cells that are within a given Cell distance in one
    direction from the given Mention.

    Note that if a candidate is passed in, all of its Mentions will be
    searched. If `directions=True``, each ngram will be returned with a
    direction in {'UP', 'DOWN', 'LEFT', 'RIGHT'}.
    Also note that if the mention is not tabular, nothing will be yielded.

    :param mention: The Mention whose neighbor Cells are being searched
    :param dist: The Cell distance within which a neighbor Cell must be to be
        considered
    :param directions: A Boolean expressing whether or not to return the
        direction of each ngram
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    :return: a *generator* of ngrams (or (ngram, direction) tuples if directions=True)
    """
    # TODO: Fix this to be more efficient (optimize with SQL query)
    spans = _to_spans(mention)
    for span in spans:
        if not span.sentence.is_tabular():
            continue

        for ngram in get_sentence_ngrams(
            span, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower
        ):
            yield ngram
            root_cell = span.sentence.cell
            for sentence in chain.from_iterable(
                [
                    _get_aligned_sentences(root_cell, "row"),
                    _get_aligned_sentences(root_cell, "col"),
                ]
            ):
                row_diff = min_row_diff([sentence, root_cell], absolute=False)
                col_diff = min_col_diff([sentence, root_cell], absolute=False)
                if (
                    row_diff ^ col_diff  # Exclusive OR
                    and abs(row_diff) + abs(col_diff) <= dist
                ):
                    if directions:
                        if col_diff == 0:
                            direction = "DOWN" if 0 < row_diff else "UP"
                        else:
                            direction = "RIGHT" if 0 < col_diff else "LEFT"
                        for ngram in tokens_to_ngrams(
                            getattr(sentence, attrib),
                            n_min=n_min,
                            n_max=n_max,
                            lower=lower,
                        ):
                            yield (ngram, direction)
                    else:
                        for ngram in tokens_to_ngrams(
                            getattr(sentence, attrib),
                            n_min=n_min,
                            n_max=n_max,
                            lower=lower,
                        ):
                            yield ngram


def get_row_ngrams(
    mention: Union[Candidate, Mention, TemporarySpanMention],
    attrib: str = "words",
    n_min: int = 1,
    n_max: int = 1,
    spread: List[int] = [0, 0],
    lower: bool = True,
) -> Iterator[str]:
    """Get the ngrams from all Cells that are in the same row as the given Mention.

    Note that if a candidate is passed in, all of its Mentions will be searched.
    Also note that if the mention is not tabular, nothing will be yielded.

    :param mention: The Mention whose row Cells are being searched
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param spread: The number of rows above and below to also consider "aligned".
    :param lower: If True, all ngrams will be returned in lower case
    """
    spans = _to_spans(mention)
    for span in spans:
        for ngram in _get_axis_ngrams(
            span,
            axis="row",
            attrib=attrib,
            n_min=n_min,
            n_max=n_max,
            spread=spread,
            lower=lower,
        ):
            yield ngram


def get_col_ngrams(
    mention: Union[Candidate, Mention, TemporarySpanMention],
    attrib: str = "words",
    n_min: int = 1,
    n_max: int = 1,
    spread: List[int] = [0, 0],
    lower: bool = True,
) -> Iterator[str]:
    """Get the ngrams from all Cells that are in the same column as the given Mention.

    Note that if a candidate is passed in, all of its Mentions will be searched.
    Also note that if the mention is not tabular, nothing will be yielded.

    :param mention: The Mention whose column Cells are being searched
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param spread: The number of cols left and right to also consider "aligned".
    :param lower: If True, all ngrams will be returned in lower case
    """
    spans = _to_spans(mention)
    for span in spans:
        for ngram in _get_axis_ngrams(
            span,
            axis="col",
            attrib=attrib,
            n_min=n_min,
            n_max=n_max,
            spread=spread,
            lower=lower,
        ):
            yield ngram


def get_aligned_ngrams(
    mention: Union[Candidate, Mention, TemporarySpanMention],
    attrib: str = "words",
    n_min: int = 1,
    n_max: int = 1,
    spread: List[int] = [0, 0],
    lower: bool = True,
) -> Iterator[str]:
    """Get the ngrams from all Cells in the same row or column as the given Mention.

    Note that if a candidate is passed in, all of its Mentions will be
    searched.
    Also note that if the mention is not tabular, nothing will be yielded.

    :param mention: The Mention whose row and column Cells are being searched
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param spread: The number of rows/cols above/below/left/right to also
        consider "aligned".
    :param lower: If True, all ngrams will be returned in lower case
    """
    spans = _to_spans(mention)
    for span in spans:
        for ngram in get_row_ngrams(
            span, attrib=attrib, n_min=n_min, n_max=n_max, spread=spread, lower=lower
        ):
            yield ngram
        for ngram in get_col_ngrams(
            span, attrib=attrib, n_min=n_min, n_max=n_max, spread=spread, lower=lower
        ):
            yield ngram


def get_head_ngrams(
    mention: Union[Candidate, Mention, TemporarySpanMention],
    axis: Optional[str] = None,
    attrib: str = "words",
    n_min: int = 1,
    n_max: int = 1,
    lower: bool = True,
) -> Iterator[str]:
    """Get the ngrams from the cell in the head of the row or column.

    More specifically, this returns the ngrams in the leftmost cell in a row and/or the
    ngrams in the topmost cell in the column, depending on the axis parameter.

    Note that if a candidate is passed in, all of its Mentions will be searched.
    Also note that if the mention is not tabular, nothing will be yielded.

    :param mention: The Mention whose head Cells are being returned
    :param axis: Which axis {'row', 'col'} to search. If None, then both row
        and col are searched.
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    """
    spans = _to_spans(mention)
    axes: Set[str] = (axis,) if axis else ("row", "col")  # type: ignore
    for span in spans:
        if span.sentence.is_tabular():
            for axis in axes:
                if getattr(span.sentence, _other_axis(axis) + "_start") == 0:
                    return
                for sentence in getattr(
                    _get_head_cell(span.sentence.cell, axis), "sentences", []
                ):
                    for ngram in tokens_to_ngrams(
                        getattr(sentence, attrib), n_min=n_min, n_max=n_max, lower=lower
                    ):
                        yield ngram


@lru_cache(maxsize=1024)
def _get_head_cell(root_cell: Cell, axis: str) -> Cell:
    other_axis = "row" if axis == "col" else "col"
    aligned_cells = _get_aligned_cells(root_cell, axis)
    return (
        sorted(aligned_cells, key=lambda x: getattr(x, other_axis + "_start"))[0]
        if aligned_cells
        else None
    )


@lru_cache(maxsize=256)
def _get_table_cells(table: Table) -> DefaultDict[Cell, List[Sentence]]:
    """Cache table cells and the cells' sentences.

    This function significantly improves the speed of `get_row_ngrams`
    primarily by reducing the number of queries that are made (which were
    previously the bottleneck. Rather than taking a single mention, then its
    sentence, then its table, then all the cells in the table, then all the
    sentences in each cell, and performing operations on that series of
    queries, this performs a single query for all the sentences in a table and
    returns all of the cells and the cells sentences directly.

    :param table: the Table object to cache.
    :return: an iterator of (Cell, [Sentence._asdict(), ...]) tuples.
    """
    sent_map: DefaultDict[Cell, List[Sentence]] = defaultdict(list)
    for sent in table.sentences:
        sent_map[sent.cell].append(sent)
    return sent_map


def _get_axis_ngrams(
    mention: Union[Candidate, Mention, TemporarySpanMention],
    axis: str,
    attrib: str = "words",
    n_min: int = 1,
    n_max: int = 1,
    spread: List[int] = [0, 0],
    lower: bool = True,
) -> Iterator[str]:
    span = _to_span(mention)

    if not span.sentence.is_tabular():
        return
        yield

    for ngram in get_sentence_ngrams(
        span, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower
    ):
        yield ngram

    for sentence in _get_aligned_sentences(span.sentence, axis, spread=spread):
        for ngram in tokens_to_ngrams(
            getattr(sentence, attrib), n_min=n_min, n_max=n_max, lower=lower
        ):
            yield ngram


@lru_cache(maxsize=1024)
def _get_aligned_cells(root_cell: Cell, axis: str) -> List[Cell]:
    aligned_cells = [
        cell
        for cell in root_cell.table.cells
        if is_axis_aligned(root_cell, cell, axis=axis) and cell != root_cell
    ]
    return aligned_cells


def _get_aligned_sentences(
    root_sentence: Sentence, axis: str, spread: List[int] = [0, 0]
) -> List[Sentence]:
    cells = _get_table_cells(root_sentence.table).items()
    aligned_sentences = [
        sentence
        for (cell, sentences) in cells
        if is_axis_aligned(root_sentence, cell, axis=axis, spread=spread)
        for sentence in sentences
        if sentence != root_sentence
    ]
    return aligned_sentences


def _other_axis(axis: str) -> str:
    return "row" if axis == "col" else "col"
