"""Fonduer tabular feature extractor."""
from typing import Dict, Iterator, List, Set, Tuple, Union

from fonduer.candidates.models import Candidate
from fonduer.candidates.models.span_mention import SpanMention, TemporarySpanMention
from fonduer.utils.config import get_config
from fonduer.utils.data_model_utils import (
    get_cell_ngrams,
    get_col_ngrams,
    get_head_ngrams,
    get_row_ngrams,
)
from fonduer.utils.utils_table import min_col_diff, min_row_diff, num_cols, num_rows

FEAT_PRE = "TAB_"
DEF_VALUE = 1

unary_tablelib_feats: Dict[str, Set] = {}
multinary_tablelib_feats: Dict[str, Set] = {}

settings = get_config()


def extract_tabular_features(
    candidates: Union[Candidate, List[Candidate]],
) -> Iterator[Tuple[int, str, int]]:
    """Extract tabular features.

    :param candidates: A list of candidates to extract features from
    """
    candidates = candidates if isinstance(candidates, list) else [candidates]
    for candidate in candidates:
        args = tuple([m.context for m in candidate.get_mentions()])
        if any(not (isinstance(arg, TemporarySpanMention)) for arg in args):
            raise ValueError(
                f"Table feature only accepts Span-type arguments, "
                f"{type(candidate)}-type found."
            )

        # Unary candidates
        if len(args) == 1:
            span = args[0]
            if span.stable_id not in unary_tablelib_feats:
                unary_tablelib_feats[span.stable_id] = set()
                for f, v in _tablelib_unary_features(span):
                    unary_tablelib_feats[span.stable_id].add((f, v))

            for f, v in unary_tablelib_feats[span.stable_id]:
                yield candidate.id, FEAT_PRE + f, v

        # Multinary candidates
        else:
            spans = args
            if any([span.sentence.is_tabular() for span in spans]):
                for i, span in enumerate(spans):
                    prefix = f"e{i}_"
                    if span.stable_id not in unary_tablelib_feats:
                        unary_tablelib_feats[span.stable_id] = set()
                        for f, v in _tablelib_unary_features(span):
                            unary_tablelib_feats[span.stable_id].add((f, v))

                    for f, v in unary_tablelib_feats[span.stable_id]:
                        yield candidate.id, FEAT_PRE + prefix + f, v

                if candidate.id not in multinary_tablelib_feats:
                    multinary_tablelib_feats[candidate.id] = set()
                    for f, v in _tablelib_multinary_features(spans):
                        multinary_tablelib_feats[candidate.id].add((f, v))

                for f, v in multinary_tablelib_feats[candidate.id]:
                    yield candidate.id, FEAT_PRE + f, v


def _tablelib_unary_features(span: SpanMention) -> Iterator[Tuple[str, int]]:
    """Table-/structure-related features for a single span."""
    if not span.sentence.is_tabular():
        return
    sentence = span.sentence
    for attrib in settings["featurization"]["tabular"]["unary_features"]["attrib"]:
        for ngram in get_cell_ngrams(
            span,
            n_max=settings["featurization"]["tabular"]["unary_features"][
                "get_cell_ngrams"
            ]["max"],
            attrib=attrib,
        ):
            yield f"CELL_{attrib.upper()}_[{ngram}]", DEF_VALUE
        for row_num in range(sentence.row_start, sentence.row_end + 1):
            yield f"ROW_NUM_[{row_num}]", DEF_VALUE
        for col_num in range(sentence.col_start, sentence.col_end + 1):
            yield f"COL_NUM_[{col_num}]", DEF_VALUE
        # NOTE: These two features could be accounted for by HTML_ATTR in
        # structural features
        yield f"ROW_SPAN_[{num_rows(sentence)}]", DEF_VALUE
        yield f"COL_SPAN_[{num_cols(sentence)}]", DEF_VALUE
        for axis in ["row", "col"]:
            for ngram in get_head_ngrams(
                span,
                axis,
                n_max=settings["featurization"]["tabular"]["unary_features"][
                    "get_head_ngrams"
                ]["max"],
                attrib=attrib,
            ):
                yield f"{axis.upper()}_HEAD_{attrib.upper()}_[{ngram}]", DEF_VALUE
        for ngram in get_row_ngrams(
            span,
            n_max=settings["featurization"]["tabular"]["unary_features"][
                "get_row_ngrams"
            ]["max"],
            attrib=attrib,
        ):
            yield f"ROW_{attrib.upper()}_[{ngram}]", DEF_VALUE
        for ngram in get_col_ngrams(
            span,
            n_max=settings["featurization"]["tabular"]["unary_features"][
                "get_col_ngrams"
            ]["max"],
            attrib=attrib,
        ):
            yield f"COL_{attrib.upper()}_[{ngram}]", DEF_VALUE
        # TODO:
        #  for ngram in get_row_ngrams(
        #      span, n_max=2, attrib=attrib, direct=False, infer=True
        #  ):
        #      yield "ROW_INFERRED_%s_[%s]" % (attrib.upper(), ngram), DEF_VALUE
        #  for ngram in get_col_ngrams(
        #      span, n_max=2, attrib=attrib, direct=False, infer=True
        #  ):
        #      yield "COL_INFERRED_%s_[%s]" % (attrib.upper(), ngram), DEF_VALUE


def _tablelib_multinary_features(
    spans: Tuple[SpanMention, ...]
) -> Iterator[Tuple[str, int]]:
    """Table-/structure-related features for multiple spans."""
    multinary_features = settings["featurization"]["tabular"]["multinary_features"]
    span_sentences = [span.sentence for span in spans]
    if all([sentence.is_tabular() for sentence in span_sentences]):
        span_tables = [sentence.table for sentence in span_sentences]
        if span_tables[1:] == span_tables[:-1]:
            yield "SAME_TABLE", DEF_VALUE
            if all([span.sentence.cell is not None for span in spans]):
                row_diff = min_row_diff(
                    span_sentences,
                    absolute=multinary_features["min_row_diff"]["absolute"],
                )
                col_diff = min_col_diff(
                    span_sentences,
                    absolute=multinary_features["min_col_diff"]["absolute"],
                )
                yield f"SAME_TABLE_ROW_DIFF_[{row_diff}]", DEF_VALUE
                yield f"SAME_TABLE_COL_DIFF_[{col_diff}]", DEF_VALUE
                yield (
                    f"SAME_TABLE_MANHATTAN_DIST_[{abs(row_diff) + abs(col_diff)}]"
                ), DEF_VALUE
                span_cells = [sentence.cell for sentence in span_sentences]
                if span_cells[1:] == span_cells[:-1]:
                    yield "SAME_CELL", DEF_VALUE
                    word_diff = sum(
                        [
                            s1.get_word_start_index() - s2.get_word_start_index()
                            for s1, s2 in zip(spans[:-1], spans[1:])
                        ]
                    )
                    yield (f"WORD_DIFF_[{word_diff}]"), DEF_VALUE
                    char_diff = sum(
                        [
                            s1.char_start - s2.char_start
                            for s1, s2 in zip(spans[:-1], spans[1:])
                        ]
                    )
                    yield (f"CHAR_DIFF_[{char_diff}]"), DEF_VALUE
                    if [span_sentences[1:] == span_sentences[:-1]]:
                        yield "SAME_SENTENCE", DEF_VALUE
        else:
            if all([sentence.cell is not None for sentence in span_sentences]):
                yield "DIFF_TABLE", DEF_VALUE
                row_diff = min_row_diff(
                    span_sentences,
                    absolute=multinary_features["min_row_diff"]["absolute"],
                )
                col_diff = min_col_diff(
                    span_sentences,
                    absolute=multinary_features["min_col_diff"]["absolute"],
                )
                yield f"DIFF_TABLE_ROW_DIFF_[{row_diff}]", DEF_VALUE
                yield f"DIFF_TABLE_COL_DIFF_[{col_diff}]", DEF_VALUE
                yield (
                    f"DIFF_TABLE_MANHATTAN_DIST_[{abs(row_diff) + abs(col_diff)}]"
                ), DEF_VALUE
