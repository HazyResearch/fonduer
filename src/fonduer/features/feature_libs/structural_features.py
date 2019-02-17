from fonduer.candidates.models.span_mention import TemporarySpanMention
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

unary_strlib_feats = {}
binary_strlib_feats = {}


def get_structural_feats(candidates):
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
                    for feature, value in strlib_unary_features(span):
                        unary_strlib_feats[span.stable_id].add((feature, value))

                for feature, value in unary_strlib_feats[span.stable_id]:
                    yield candidate.id, FEATURE_PREFIX + feature, value

        # Binary candidates
        elif len(args) == 2:
            span1, span2 = args
            if span1.sentence.is_structural() or span2.sentence.is_structural():
                for span, prefix in [(span1, "e1_"), (span2, "e2_")]:
                    if span.stable_id not in unary_strlib_feats:
                        unary_strlib_feats[span.stable_id] = set()
                        for feature, value in strlib_unary_features(span):
                            unary_strlib_feats[span.stable_id].add((feature, value))

                    for feature, value in unary_strlib_feats[span.stable_id]:
                        yield candidate.id, FEATURE_PREFIX + prefix + feature, value

                if candidate.id not in binary_strlib_feats:
                    binary_strlib_feats[candidate.id] = set()
                    for feature, value in strlib_binary_features(span1, span2):
                        binary_strlib_feats[candidate.id].add((feature, value))

                for feature, value in binary_strlib_feats[candidate.id]:
                    yield candidate.id, FEATURE_PREFIX + feature, value
        else:
            raise NotImplementedError(
                "Only handles unary and binary candidates currently"
            )


def strlib_unary_features(span):
    """
    Structural-related features for a single span
    """
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


def strlib_binary_features(span1, span2):
    """
    Structural-related features for a pair of spans
    """
    yield f"COMMON_ANCESTOR_[{' '.join(common_ancestor((span1, span2)))}]", DEF_VALUE

    yield (
        f"LOWEST_ANCESTOR_DEPTH_[" f"{lowest_common_ancestor_depth((span1, span2))}]"
    ), DEF_VALUE
