############################
# Tabular modality helpers
############################

from builtins import range
from itertools import chain

from fonduer.candidates.models import TemporarySpan
from fonduer.parser.models import Sentence
from fonduer.supervision.lf_helpers.textual import get_left_ngrams, get_right_ngrams
from fonduer.utils import tokens_to_ngrams
from fonduer.utils.utils_table import (
    is_axis_aligned,
    is_col_aligned,
    is_row_aligned,
    min_col_diff,
    min_row_diff,
)


def same_document(c):
    """Return True if all Spans in the given candidate are from the same Document.

    :param c: The candidate whose Spans are being compared
    :rtype: boolean
    """
    return all(
        c[i].sentence.document is not None
        and c[i].sentence.document == c[0].sentence.document
        for i in range(len(c))
    )


def same_table(c):
    """Return True if all Spans in the given candidate are from the same Table.

    :param c: The candidate whose Spans are being compared
    :rtype: boolean
    """
    return all(
        c[i].sentence.is_tabular() and c[i].sentence.table == c[0].sentence.table
        for i in range(len(c))
    )


def same_row(c):
    """Return True if all Spans in the given candidate are from the same Row.

    :param c: The candidate whose Spans are being compared
    :rtype: boolean
    """
    return same_table(c) and all(
        is_row_aligned(c[i].sentence, c[0].sentence) for i in range(len(c))
    )


def same_col(c):
    """Return True if all Spans in the given candidate are from the same Col.

    :param c: The candidate whose Spans are being compared
    :rtype: boolean
    """
    return same_table(c) and all(
        is_col_aligned(c[i].sentence, c[0].sentence) for i in range(len(c))
    )


def is_tabular_aligned(c):
    """Return True if all Spans in the given candidate are from the same Row or Col.

    :param c: The candidate whose Spans are being compared
    :rtype: boolean
    """
    return same_table(c) and (
        is_col_aligned(c[i].sentence, c[0].sentence)
        or is_row_aligned(c[i].sentence, c[0].sentence)
        for i in range(len(c))
    )


def same_cell(c):
    """Return True if all Spans in the given candidate are from the same Cell.

    :param c: The candidate whose Spans are being compared
    :rtype: boolean
    """
    return all(
        c[i].sentence.cell is not None and c[i].sentence.cell == c[0].sentence.cell
        for i in range(len(c))
    )


def same_sentence(c):
    """Return True if all Spans in the given candidate are from the same Sentence.

    :param c: The candidate whose Spans are being compared
    :rtype: boolean
    """
    return all(
        c[i].sentence is not None and c[i].sentence == c[0].sentence
        for i in range(len(c))
    )


def get_max_col_num(span):
    """Return the largest column number that a Span occupies.

    :param span: The Span to evaluate. If a candidate is given, default to its last Span.
    :rtype: integer or None
    """
    span = span if isinstance(span, TemporarySpan) else span[-1]
    if span.sentence.is_tabular():
        return span.sentence.cell.col_end
    else:
        return None


def get_min_col_num(span):
    """Return the lowest column number that a Span occupies.

    :param span: The Span to evaluate. If a candidate is given, default to its first Span.
    :rtype: integer or None
    """
    span = span if isinstance(span, TemporarySpan) else span[0]
    if span.sentence.is_tabular():
        return span.sentence.cell.col_start
    else:
        return None


def get_sentence_ngrams(span, attrib="words", n_min=1, n_max=1, lower=True):
    """Get the ngrams that are in the Sentence of the given span, not including itself.

    Note that if a candidate is passed in, all of its Spans will be searched.

    :param span: The Span whose Sentence is being searched
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    :rtype: a *generator* of ngrams
    """
    spans = [span] if isinstance(span, TemporarySpan) else span.get_contexts()
    for span in spans:
        for ngram in get_left_ngrams(
            span, window=100, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower
        ):
            yield ngram
        for ngram in get_right_ngrams(
            span, window=100, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower
        ):
            yield ngram


def get_neighbor_sentence_ngrams(
    span, d=1, attrib="words", n_min=1, n_max=1, lower=True
):
    """Get the ngrams that are in the neighoring Sentences of the given Span.

    Note that if a candidate is passed in, all of its Spans will be searched.

    :param span: The span whose neighbor Sentences are being searched
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    :rtype: a *generator* of ngrams
    """
    spans = [span] if isinstance(span, TemporarySpan) else span.get_contexts()
    for span in spans:
        for ngram in chain.from_iterable(
            [
                tokens_to_ngrams(
                    getattr(sentence, attrib), n_min=n_min, n_max=n_max, lower=lower
                )
                for sentence in span.sentence.document.sentences
                if abs(sentence.sentence_num - span.sentence.sentence_num) <= d
                and sentence != span.sentence
            ]
        ):
            yield ngram


def get_cell_ngrams(span, attrib="words", n_min=1, n_max=1, lower=True):
    """Get the ngrams that are in the Cell of the given span, not including itself.

    Note that if a candidate is passed in, all of its Spans will be searched.

    :param span: The span whose Cell is being searched
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    :rtype: a *generator* of ngrams
    """
    spans = [span] if isinstance(span, TemporarySpan) else span.get_contexts()
    for span in spans:
        for ngram in get_sentence_ngrams(
            span, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower
        ):
            yield ngram
        if isinstance(span.sentence, Sentence) and span.sentence.cell is not None:
            for ngram in chain.from_iterable(
                [
                    tokens_to_ngrams(
                        getattr(sentence, attrib), n_min=n_min, n_max=n_max, lower=lower
                    )
                    for sentence in span.sentence.cell.sentences
                    if sentence != span.sentence
                ]
            ):
                yield ngram


def get_neighbor_cell_ngrams(
    span, dist=1, directions=False, attrib="words", n_min=1, n_max=1, lower=True
):
    """Get the ngrams from all Cells that are within a given Cell distance in one direction from the given Span.

    Note that if a candidate is passed in, all of its Spans will be searched.
    If `directions=True``, each ngram will be returned with a direction in {'UP', 'DOWN', 'LEFT', 'RIGHT'}.

    :param span: The span whose neighbor Cells are being searched
    :param dist: The Cell distance within which a neighbor Cell must be to be considered
    :param directions: A Boolean expressing whether or not to return the direction of each ngram
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    :rtype: a *generator* of ngrams (or (ngram, direction) tuples if directions=True)
    """
    # TODO: Fix this to be more efficient (optimize with SQL query)
    spans = [span] if isinstance(span, TemporarySpan) else span.get_contexts()
    for span in spans:
        for ngram in get_sentence_ngrams(
            span, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower
        ):
            yield ngram
        if isinstance(span.sentence, Sentence) and span.sentence.cell is not None:
            root_cell = span.sentence.cell
            for sentence in chain.from_iterable(
                [
                    _get_aligned_sentences(root_cell, "row"),
                    _get_aligned_sentences(root_cell, "col"),
                ]
            ):
                row_diff = min_row_diff(sentence, root_cell, absolute=False)
                col_diff = min_col_diff(sentence, root_cell, absolute=False)
                if (
                    (row_diff or col_diff)
                    and not (row_diff and col_diff)
                    and abs(row_diff) + abs(col_diff) <= dist
                ):
                    if directions:
                        direction = ""
                        if col_diff == 0:
                            if 0 < row_diff and row_diff <= dist:
                                direction = "UP"
                            elif 0 > row_diff and row_diff >= -dist:
                                direction = "DOWN"
                        elif row_diff == 0:
                            if 0 < col_diff and col_diff <= dist:
                                direction = "RIGHT"
                            elif 0 > col_diff and col_diff >= -dist:
                                direction = "LEFT"
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


def get_row_ngrams(span, attrib="words", n_min=1, n_max=1, spread=[0, 0], lower=True):
    """Get the ngrams from all Cells that are in the same row as the given Span.

    Note that if a candidate is passed in, all of its Spans will be searched.

    :param span: The span whose row Cells are being searched
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    :rtype: a *generator* of ngrams
    """
    spans = [span] if isinstance(span, TemporarySpan) else span.get_contexts()
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


def get_col_ngrams(span, attrib="words", n_min=1, n_max=1, spread=[0, 0], lower=True):
    """Get the ngrams from all Cells that are in the same column as the given Span.

    Note that if a candidate is passed in, all of its Spans will be searched.

    :param span: The span whose column Cells are being searched
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    :rtype: a *generator* of ngrams
    """
    spans = [span] if isinstance(span, TemporarySpan) else span.get_contexts()
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
    span, attrib="words", n_min=1, n_max=1, spread=[0, 0], lower=True
):
    """Get the ngrams from all Cells that are in the same row or column as the given Span.

    Note that if a candidate is passed in, all of its Spans will be searched.

    :param span: The span whose row and column Cells are being searched
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    :rtype: a *generator* of ngrams
    """
    spans = [span] if isinstance(span, TemporarySpan) else span.get_contexts()
    for span in spans:
        for ngram in get_row_ngrams(
            span, attrib=attrib, n_min=n_min, n_max=n_max, spread=spread, lower=lower
        ):
            yield ngram
        for ngram in get_col_ngrams(
            span, attrib=attrib, n_min=n_min, n_max=n_max, spread=spread, lower=lower
        ):
            yield ngram


def get_head_ngrams(span, axis=None, attrib="words", n_min=1, n_max=1, lower=True):
    """Get the ngrams from the cell in the head of the row or column.

    More specifically, this returns the ngrams in the leftmost cell in a row and/or the
    ngrams in the topmost cell in the column, depending on the axis parameter.

    Note that if a candidate is passed in, all of its Spans will be searched.

    :param span: The span whose head Cells are being returned
    :param axis: Which axis {'row', 'col'} to search. If None, then both row and col are searched.
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    :rtype: a *generator* of ngrams
    """
    spans = [span] if isinstance(span, TemporarySpan) else span.get_contexts()
    axes = [axis] if axis else ["row", "col"]
    for span in spans:
        if not span.sentence.cell:
            return
        else:
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


def _get_head_cell(root_cell, axis):
    other_axis = "row" if axis == "col" else "col"
    aligned_cells = _get_aligned_cells(root_cell, axis)
    return (
        sorted(aligned_cells, key=lambda x: getattr(x, other_axis + "_start"))[0]
        if aligned_cells
        else []
    )


def _get_axis_ngrams(
    span, axis, attrib="words", n_min=1, n_max=1, spread=[0, 0], lower=True
):
    for ngram in get_sentence_ngrams(
        span, attrib=attrib, n_min=n_min, n_max=n_max, lower=lower
    ):
        yield ngram
    if span.sentence.cell is not None:
        for sentence in _get_aligned_sentences(span.sentence, axis, spread=spread):
            for ngram in tokens_to_ngrams(
                getattr(sentence, attrib), n_min=n_min, n_max=n_max, lower=lower
            ):
                yield ngram


def _get_aligned_cells(root_cell, axis):
    aligned_cells = [
        cell
        for cell in root_cell.table.cells
        if is_axis_aligned(root_cell, cell, axis=axis) and cell != root_cell
    ]
    return aligned_cells


def _get_aligned_sentences(root_sentence, axis, spread=[0, 0]):
    return [
        sentence
        for cell in root_sentence.table.cells
        if is_axis_aligned(root_sentence, cell, axis=axis, spread=spread)
        for sentence in cell.sentences
        if sentence != root_sentence
    ]


def _other_axis(axis):
    return "row" if axis == "col" else "col"
