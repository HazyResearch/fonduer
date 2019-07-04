from fonduer.candidates.models import ImplicitSpanMention
from fonduer.candidates.models.span_mention import TemporarySpanMention

FEAT_PRE = "CORE_"
DEF_VALUE = 1

unary_feats = {}


def get_core_feats(candidates):
    candidates = candidates if isinstance(candidates, list) else [candidates]
    for candidate in candidates:
        args = tuple([m.context for m in candidate.get_mentions()])
        if any(not (isinstance(arg, TemporarySpanMention)) for arg in args):
            raise ValueError(
                f"Core feature only accepts Span-type arguments, "
                f"{type(candidate)}-type found."
            )

        # Unary candidates
        if len(args) == 1:
            span = args[0]

            if span.stable_id not in unary_feats:
                unary_feats[span.stable_id] = set()
                for f in _generate_core_feats(span):
                    unary_feats[span.stable_id].add(f)

            for f in unary_feats[span.stable_id]:
                yield candidate.id, FEAT_PRE + f, DEF_VALUE

        # Binary candidates
        elif len(args) == 2:
            span1, span2 = args
            for span, pre in [(span1, "e1_"), (span2, "e2_")]:
                if span.stable_id not in unary_feats:
                    unary_feats[span.stable_id] = set()
                    for f in _generate_core_feats(span):
                        unary_feats[span.stable_id].add(f)

                for f in unary_feats[span.stable_id]:
                    yield candidate.id, FEAT_PRE + pre + f, DEF_VALUE
        else:
            raise NotImplementedError(
                "Only handles unary and binary candidates currently"
            )


def _generate_core_feats(span):
    yield (
        f"SPAN_TYPE_["
        f"{'IMPLICIT' if isinstance(span, ImplicitSpanMention) else 'EXPLICIT'}"
        f"]"
    )

    if span.get_span()[0].isupper():
        yield "STARTS_WITH_CAPITAL"

    yield f"LENGTH_{span.get_num_words()}"
