"""Fonduer structural feature extractor."""
from typing import Dict, Iterator, List, Set, Tuple, Union

from fonduer.candidates.models import Candidate
from fonduer.candidates.models.span_mention import SpanMention, TemporarySpanMention
from fonduer.utils.data_model_utils import (
    common_ancestor,
    get_ancestor_class_names,
    get_ancestor_id_names,
    get_ancestor_tag_names,
    get_attributes,
    get_next_sibling_tags,
    get_parent_tag,
    get_prev_sibling_tags,
    get_tag,
    lowest_common_ancestor_depth,
)

FEATURE_PREFIX = "STR_"
DEF_VALUE = 1

unary_strlib_feats: Dict[str, Set[Tuple[str, int]]] = {}
multinary_strlib_feats: Dict[str, Set[Tuple[str, int]]] = {}


def extract_structural_features(
    candidates: Union[Candidate, List[Candidate]],
) -> Iterator[Tuple[int, str, int]]:
    """Extract structural features.

    :param candidates: A list of candidates to extract features from
    """
    candidates = candidates if isinstance(candidates, list) else [candidates]
    for candidate in candidates:
        args = tuple([m.context for m in candidate.get_mentions()])
        if any(not (isinstance(arg, TemporarySpanMention)) for arg in args):
            raise ValueError(
                f"Structural feature only accepts Span-type arguments, "
                f"{type(candidate)}-type found."
            )

        # Unary candidates
        if len(args) == 1:
            span = args[0]
            if span.sentence.is_structural():
                if span.stable_id not in unary_strlib_feats:
                    unary_strlib_feats[span.stable_id] = set()
                    for feature, value in _strlib_unary_features(span):
                        unary_strlib_feats[span.stable_id].add((feature, value))

                for feature, value in unary_strlib_feats[span.stable_id]:
                    yield candidate.id, FEATURE_PREFIX + feature, value

        # Multinary candidates
        else:
            spans = args
            if all([span.sentence.is_structural() for span in spans]):
                for i, span in enumerate(spans):
                    prefix = f"e{i}_"
                    if span.stable_id not in unary_strlib_feats:
                        unary_strlib_feats[span.stable_id] = set()
                        for feature, value in _strlib_unary_features(span):
                            unary_strlib_feats[span.stable_id].add((feature, value))

                    for feature, value in unary_strlib_feats[span.stable_id]:
                        yield candidate.id, FEATURE_PREFIX + prefix + feature, value

                if candidate.id not in multinary_strlib_feats:
                    multinary_strlib_feats[candidate.id] = set()
                    for feature, value in _strlib_multinary_features(spans):
                        multinary_strlib_feats[candidate.id].add((feature, value))

                for feature, value in multinary_strlib_feats[candidate.id]:
                    yield candidate.id, FEATURE_PREFIX + feature, value


def _strlib_unary_features(span: SpanMention) -> Iterator[Tuple[str, int]]:
    """Structural-related features for a single span."""
    if not span.sentence.is_structural():
        return

    yield f"TAG_{get_tag(span)}", DEF_VALUE

    for attr in get_attributes(span):
        yield f"HTML_ATTR_{attr}", DEF_VALUE

    yield f"PARENT_TAG_{get_parent_tag(span)}", DEF_VALUE

    prev_tags = get_prev_sibling_tags(span)
    if len(prev_tags):
        yield f"PREV_SIB_TAG_{prev_tags[-1]}", DEF_VALUE
        yield f"NODE_POS_{len(prev_tags) + 1}", DEF_VALUE
    else:
        yield "FIRST_NODE", DEF_VALUE

    next_tags = get_next_sibling_tags(span)
    if len(next_tags):
        yield f"NEXT_SIB_TAG_{next_tags[0]}", DEF_VALUE
    else:
        yield "LAST_NODE", DEF_VALUE

    yield f"ANCESTOR_CLASS_[{' '.join(get_ancestor_class_names(span))}]", DEF_VALUE

    yield f"ANCESTOR_TAG_[{' '.join(get_ancestor_tag_names(span))}]", DEF_VALUE

    yield f"ANCESTOR_ID_[{' '.join(get_ancestor_id_names(span))}]", DEF_VALUE


def _strlib_multinary_features(
    spans: Tuple[SpanMention, ...]
) -> Iterator[Tuple[str, int]]:
    """Structural-related features for multiple spans."""
    yield f"COMMON_ANCESTOR_[{' '.join(common_ancestor(spans))}]", DEF_VALUE

    yield (
        f"LOWEST_ANCESTOR_DEPTH_[" f"{lowest_common_ancestor_depth(spans)}]"
    ), DEF_VALUE
