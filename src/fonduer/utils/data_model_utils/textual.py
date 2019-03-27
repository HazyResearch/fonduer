############################
# Textual modality utilities
############################

from fonduer.utils.data_model_utils.utils import _to_span
from fonduer.utils.utils import tokens_to_ngrams


def get_between_ngrams(c, attrib="words", n_min=1, n_max=1, lower=True):
    """Return the ngrams *between* two unary Mentions of a binary-Mention Candidate.

    Get the ngrams *between* two unary Mentions of a binary-Mention Candidate,
    where both share the same sentence Context.

    :param c: The binary-Mention Candidate to evaluate.
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If 'True', all ngrams will be returned in lower case
    :rtype: a *generator* of ngrams
    """
    if len(c) != 2:
        raise ValueError("Only applicable to binary Candidates")
    span0 = _to_span(c[0])
    span1 = _to_span(c[1])
    if span0.sentence != span1.sentence:
        raise ValueError(
            "Only applicable to Candidates where both spans are \
                          from the same immediate Context."
        )
    distance = abs(span0.get_word_start_index() - span1.get_word_start_index())
    if span0.get_word_start_index() < span1.get_word_start_index():
        for ngram in get_right_ngrams(
            span0,
            window=distance - 1,
            attrib=attrib,
            n_min=n_min,
            n_max=n_max,
            lower=lower,
        ):
            yield ngram
    else:  # span0.get_word_start_index() > span1.get_word_start_index()
        for ngram in get_right_ngrams(
            span1,
            window=distance - 1,
            attrib=attrib,
            n_min=n_min,
            n_max=n_max,
            lower=lower,
        ):
            yield ngram


def get_left_ngrams(mention, window=3, attrib="words", n_min=1, n_max=1, lower=True):
    """Get the ngrams within a window to the *left* from the sentence Context.

    For higher-arity Candidates, defaults to the *first* argument.

    :param mention: The Mention to evaluate. If a candidate is given, default
        to its first Mention.
    :param window: The number of tokens to the left of the first argument to
        return.
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    :rtype: a *generator* of ngrams
    """
    span = _to_span(mention)
    i = span.get_word_start_index()
    for ngram in tokens_to_ngrams(
        getattr(span.sentence, attrib)[max(0, i - window) : i],
        n_min=n_min,
        n_max=n_max,
        lower=lower,
    ):
        yield ngram


def get_right_ngrams(mention, window=3, attrib="words", n_min=1, n_max=1, lower=True):
    """Get the ngrams within a window to the *right* from the sentence Context.

    For higher-arity Candidates, defaults to the *last* argument.

    :param mention: The Mention to evaluate. If a candidate is given, default
        to its last Mention.
    :param window: The number of tokens to the left of the first argument to
        return
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    :rtype: a *generator* of ngrams
    """
    span = _to_span(mention, idx=-1)
    i = span.get_word_end_index()
    for ngram in tokens_to_ngrams(
        getattr(span.sentence, attrib)[i + 1 : i + 1 + window],
        n_min=n_min,
        n_max=n_max,
        lower=lower,
    ):
        yield ngram
