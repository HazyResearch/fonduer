"""Fonduer textual feature extractor."""
from builtins import range
from typing import Any, Callable, Dict, Iterator, List, Set, Tuple, Union

from treedlib import (
    Children,
    Compile,
    Indicator,
    LeftNgrams,
    LeftSiblings,
    Mention,
    Ngrams,
    Parents,
    RightNgrams,
    RightSiblings,
    compile_relation_feature_generator,
)

from fonduer.candidates.models import Candidate, ImplicitSpanMention, SpanMention
from fonduer.features.feature_libs.tree_structs import corenlp_to_xmltree
from fonduer.utils.config import get_config
from fonduer.utils.data_model_utils import get_left_ngrams, get_right_ngrams
from fonduer.utils.utils import get_as_dict, tokens_to_ngrams

DEF_VALUE = 1

unary_ddlib_feats: Dict[str, Set] = {}
unary_word_feats: Dict[str, Set] = {}
unary_tdl_feats: Dict[str, Set] = {}
multinary_tdl_feats: Dict[str, Set] = {}
settings = get_config()


def extract_textual_features(
    candidates: Union[Candidate, List[Candidate]],
) -> Iterator[Tuple[int, str, int]]:
    """Extract textual features.

    :param candidates: A list of candidates to extract features from
    """
    candidates = candidates if isinstance(candidates, list) else [candidates]
    for candidate in candidates:
        args = tuple([m.context for m in candidate.get_mentions()])
        if not (isinstance(args[0], (SpanMention, ImplicitSpanMention))):
            raise ValueError(
                f"Accepts Span/ImplicitSpan-type mentions, {type(args[0])}-type found."
            )

        # Unary candidates
        if len(args) == 1:
            span: Union[SpanMention, ImplicitSpanMention] = args[0]
            if span.sentence.is_lingual():
                get_tdl_feats = _compile_entity_feature_generator()
                xmltree = corenlp_to_xmltree(span.sentence)
                sidxs = list(
                    range(span.get_word_start_index(), span.get_word_end_index() + 1)
                )
                if len(sidxs) > 0:
                    # Add DDLIB entity features
                    for f in _get_ddlib_feats(span, get_as_dict(span.sentence), sidxs):
                        yield candidate.id, f"DDL_{f}", DEF_VALUE
                    # Add TreeDLib entity features
                    if span.stable_id not in unary_tdl_feats:
                        unary_tdl_feats[span.stable_id] = set()
                        for f in get_tdl_feats(xmltree.root, sidxs):
                            unary_tdl_feats[span.stable_id].add(f)
                    for f in unary_tdl_feats[span.stable_id]:
                        yield candidate.id, f"TDL_{f}", DEF_VALUE
            for f in _get_word_feats(span):
                yield candidate.id, f"BASIC_{f}", DEF_VALUE

        # Multinary candidates
        else:
            spans = args
            if all([span.sentence.is_lingual() for span in spans]):
                get_tdl_feats = compile_relation_feature_generator(is_multary=True)
                sents = [get_as_dict(span.sentence) for span in spans]
                xmltree = corenlp_to_xmltree(spans[0].sentence)
                s_idxs = [
                    list(
                        range(
                            span.get_word_start_index(), span.get_word_end_index() + 1
                        )
                    )
                    for span in spans
                ]
                if all([len(s_idx) > 0 for s_idx in s_idxs]):

                    # Add DDLIB entity features for relation
                    for span, sent, s_idx, i in zip(
                        spans, sents, s_idxs, range(len(spans))
                    ):

                        for f in _get_ddlib_feats(span, sent, s_idx):
                            yield candidate.id, f"DDL_e{i}_{f}", DEF_VALUE

                    # Add TreeDLib relation features
                    if candidate.id not in multinary_tdl_feats:
                        multinary_tdl_feats[candidate.id] = set()
                        for f in get_tdl_feats(xmltree.root, s_idxs):
                            multinary_tdl_feats[candidate.id].add(f)
                    for f in multinary_tdl_feats[candidate.id]:
                        yield candidate.id, f"TDL_{f}", DEF_VALUE
            for i, span in enumerate(spans):
                for f in _get_word_feats(span):
                    yield candidate.id, f"BASIC_e{i}_{f}", DEF_VALUE


def _compile_entity_feature_generator() -> Callable:
    """Compile entity feature generator.

    Given optional arguments, returns a generator function which accepts an xml
    root and a list of indexes for a mention, and will generate relation
    features for this entity.
    """
    BASIC_ATTRIBS_REL = ["lemma", "dep_label"]

    m = Mention(0)

    # Basic relation feature templates
    temps = [
        [Indicator(m, a) for a in BASIC_ATTRIBS_REL],
        Indicator(m, "dep_label,lemma"),
        # The *first element on the* path to the root: ngram lemmas along it
        Ngrams(Parents(m, 3), "lemma", (1, 3)),
        Ngrams(Children(m), "lemma", (1, 3)),
        # The siblings of the mention
        [LeftNgrams(LeftSiblings(m), a) for a in BASIC_ATTRIBS_REL],
        [RightNgrams(RightSiblings(m), a) for a in BASIC_ATTRIBS_REL],
    ]

    # return generator function
    return Compile(temps).apply_mention


def _get_ddlib_feats(
    span: SpanMention, context: Dict[str, Any], idxs: List[int]
) -> Iterator[str]:
    """Minimalist port of generic mention features from ddlib."""
    if span.stable_id not in unary_ddlib_feats:
        unary_ddlib_feats[span.stable_id] = set()

        for seq_feat in _get_seq_features(context, idxs):
            unary_ddlib_feats[span.stable_id].add(seq_feat)

        for window_feat in _get_window_features(context, idxs):
            unary_ddlib_feats[span.stable_id].add(window_feat)

    for f in unary_ddlib_feats[span.stable_id]:
        yield f


def _get_seq_features(context: Dict[str, Any], idxs: List[int]) -> Iterator[str]:
    yield f"WORD_SEQ_[{' '.join(context['words'][i] for i in idxs)}]"
    yield f"LEMMA_SEQ_[{' '.join(context['lemmas'][i] for i in idxs)}]"
    yield f"POS_SEQ_[{' '.join(context['pos_tags'][i] for i in idxs)}]"
    yield f"DEP_SEQ_[{' '.join(context['dep_labels'][i] for i in idxs)}]"


def _get_window_features(
    context: Dict[str, Any],
    idxs: List[int],
    window: int = settings["featurization"]["textual"]["window_feature"]["size"],
    combinations: bool = settings["featurization"]["textual"]["window_feature"][
        "combinations"
    ],
    isolated: bool = settings["featurization"]["textual"]["window_feature"]["isolated"],
) -> Iterator[str]:
    left_lemmas = []
    left_pos_tags = []
    right_lemmas = []
    right_pos_tags = []
    try:
        for i in range(1, window + 1):
            lemma = context["lemmas"][idxs[0] - i]
            try:
                float(lemma)
                lemma = "_NUMBER"
            except ValueError:
                pass
            left_lemmas.append(lemma)
            left_pos_tags.append(context["pos_tags"][idxs[0] - i])
    except IndexError:
        pass
    left_lemmas.reverse()
    left_pos_tags.reverse()
    try:
        for i in range(1, window + 1):
            lemma = context["lemmas"][idxs[-1] + i]
            try:
                float(lemma)
                lemma = "_NUMBER"
            except ValueError:
                pass
            right_lemmas.append(lemma)
            right_pos_tags.append(context["pos_tags"][idxs[-1] + i])
    except IndexError:
        pass
    if isolated:
        for i in range(len(left_lemmas)):
            yield f"W_LEFT_{i + 1}_[{' '.join(left_lemmas[-i - 1 :])}]"
            yield f"W_LEFT_POS_{i + 1}_[{' '.join(left_pos_tags[-i - 1 :])}]"
        for i in range(len(right_lemmas)):
            yield f"W_RIGHT_{i + 1}_[{' '.join(right_lemmas[: i + 1])}]"
            yield f"W_RIGHT_POS_{i + 1}_[{' '.join(right_pos_tags[: i + 1])}]"
    if combinations:
        for i in range(len(left_lemmas)):
            curr_left_lemmas = " ".join(left_lemmas[-i - 1 :])
            try:
                curr_left_pos_tags = " ".join(left_pos_tags[-i - 1 :])
            except TypeError:
                new_pos_tags = []
                for pos in left_pos_tags[-i - 1 :]:
                    to_add = pos
                    if not to_add:
                        to_add = "None"
                    new_pos_tags.append(to_add)
                curr_left_pos_tags = " ".join(new_pos_tags)
            for j in range(len(right_lemmas)):
                curr_right_lemmas = " ".join(right_lemmas[: j + 1])
                try:
                    curr_right_pos_tags = " ".join(right_pos_tags[: j + 1])
                except TypeError:
                    new_pos_tags = []
                    for pos in right_pos_tags[: j + 1]:
                        to_add = pos
                        if not to_add:
                            to_add = "None"
                        new_pos_tags.append(to_add)
                    curr_right_pos_tags = " ".join(new_pos_tags)
                yield (
                    f"W_LEMMA_L_{i + 1}_R_{j + 1}_"
                    f"[{curr_left_lemmas}]_[{curr_right_lemmas}]"
                )
                yield (
                    f"W_POS_L_{i + 1}_R_{j + 1}_"
                    f"[{curr_left_pos_tags}]_[{curr_right_pos_tags}]"
                )


def _get_word_feats(span: SpanMention) -> Iterator[str]:
    attrib = "words"

    if span.stable_id not in unary_word_feats:
        unary_word_feats[span.stable_id] = set()

        for ngram in tokens_to_ngrams(span.get_attrib_tokens(attrib), n_min=1, n_max=2):
            feature = f"CONTAINS_{attrib.upper()}_[{ngram}]"
            unary_word_feats[span.stable_id].add(feature)

        for ngram in get_left_ngrams(
            span,
            window=settings["featurization"]["textual"]["word_feature"]["window"],
            n_max=2,
            attrib=attrib,
        ):
            feature = f"LEFT_{attrib.upper()}_[{ngram}]"
            unary_word_feats[span.stable_id].add(feature)

        for ngram in get_right_ngrams(
            span,
            window=settings["featurization"]["textual"]["word_feature"]["window"],
            n_max=2,
            attrib=attrib,
        ):
            feature = f"RIGHT_{attrib.upper()}_[{ngram}]"
            unary_word_feats[span.stable_id].add(feature)

        unary_word_feats[span.stable_id].add(
            (
                f"SPAN_TYPE_["
                f"{'IMPLICIT' if isinstance(span, ImplicitSpanMention) else 'EXPLICIT'}"
                f"]"
            )
        )

        if span.get_span()[0].isupper():
            unary_word_feats[span.stable_id].add("STARTS_WITH_CAPITAL")

        unary_word_feats[span.stable_id].add(f"LENGTH_{span.get_num_words()}")

    for f in unary_word_feats[span.stable_id]:
        yield f
