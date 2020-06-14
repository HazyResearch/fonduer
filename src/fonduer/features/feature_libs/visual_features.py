"""Fonduer visual feature extractor."""
from typing import Dict, Iterator, List, Set, Tuple, Union

from fonduer.candidates.models import Candidate
from fonduer.candidates.models.span_mention import SpanMention, TemporarySpanMention
from fonduer.utils.data_model_utils import (
    get_visual_aligned_lemmas,
    is_horz_aligned,
    is_vert_aligned,
    is_vert_aligned_center,
    is_vert_aligned_left,
    is_vert_aligned_right,
    same_page,
)

FEAT_PRE = "VIZ_"
DEF_VALUE = 1

unary_vizlib_feats: Dict[str, Set] = {}
multinary_vizlib_feats: Dict[str, Set] = {}


def extract_visual_features(
    candidates: Union[Candidate, List[Candidate]],
) -> Iterator[Tuple[int, str, int]]:
    """Extract visual features.

    :param candidates: A list of candidates to extract features from
    """
    candidates = candidates if isinstance(candidates, list) else [candidates]
    for candidate in candidates:
        args = tuple([m.context for m in candidate.get_mentions()])
        if any(not (isinstance(arg, TemporarySpanMention)) for arg in args):
            raise ValueError(
                f"Visual feature only accepts Span-type arguments, "
                f"{type(candidate)}-type found."
            )

        # Unary candidates
        if len(args) == 1:
            span = args[0]
            # Add VisualLib entity features (if applicable)
            if span.sentence.is_visual():
                if span.stable_id not in unary_vizlib_feats:
                    unary_vizlib_feats[span.stable_id] = set()
                    for f, v in _vizlib_unary_features(span):
                        unary_vizlib_feats[span.stable_id].add((f, v))

                for f, v in unary_vizlib_feats[span.stable_id]:
                    yield candidate.id, FEAT_PRE + f, v

        # Multinary candidates
        else:
            spans = args
            # Add VisualLib entity features (if applicable)
            if all([span.sentence.is_visual() for span in spans]):
                for i, span in enumerate(spans):
                    prefix = f"e{i}_"
                    if span.stable_id not in unary_vizlib_feats:
                        unary_vizlib_feats[span.stable_id] = set()
                        for f, v in _vizlib_unary_features(span):
                            unary_vizlib_feats[span.stable_id].add((f, v))

                    for f, v in unary_vizlib_feats[span.stable_id]:
                        yield candidate.id, FEAT_PRE + prefix + f, v

                if candidate.id not in multinary_vizlib_feats:
                    multinary_vizlib_feats[candidate.id] = set()
                    for f, v in _vizlib_multinary_features(spans):
                        multinary_vizlib_feats[candidate.id].add((f, v))

                for f, v in multinary_vizlib_feats[candidate.id]:
                    yield candidate.id, FEAT_PRE + f, v


def _vizlib_unary_features(span: SpanMention) -> Iterator[Tuple[str, int]]:
    """Visual-related features for a single span."""
    if not span.sentence.is_visual():
        return

    for f in get_visual_aligned_lemmas(span):
        yield f"ALIGNED_{f}", DEF_VALUE

    for page in set(span.get_attrib_tokens("page")):
        yield f"PAGE_[{page}]", DEF_VALUE


def _vizlib_multinary_features(
    spans: Tuple[SpanMention, ...]
) -> Iterator[Tuple[str, int]]:
    """Visual-related features for multiple spans."""
    if same_page(spans):
        yield "SAME_PAGE", DEF_VALUE

        if is_horz_aligned(spans):
            yield "HORZ_ALIGNED", DEF_VALUE

        if is_vert_aligned(spans):
            yield "VERT_ALIGNED", DEF_VALUE

        if is_vert_aligned_left(spans):
            yield "VERT_ALIGNED_LEFT", DEF_VALUE

        if is_vert_aligned_right(spans):
            yield "VERT_ALIGNED_RIGHT", DEF_VALUE

        if is_vert_aligned_center(spans):
            yield "VERT_ALIGNED_CENTER", DEF_VALUE
