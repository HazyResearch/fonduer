"""Fonduer textual modality utilities."""
from itertools import chain
from typing import Iterator, Union

from fonduer.candidates.models import Candidate, Mention
from fonduer.candidates.models.span_mention import TemporarySpanMention
from fonduer.utils.data_model_utils.utils import _to_span, _to_spans
from fonduer.utils.utils import tokens_to_ngrams


def same_sentence(c: Candidate) -> bool:
    """Return True if all Mentions in the given candidate are from the same Sentence.

    :param c: The candidate whose Mentions are being compared
    """
    return all(
        _to_span(c[i]).sentence is not None
        and _to_span(c[i]).sentence == _to_span(c[0]).sentence
        for i in range(len(c))
    )


def get_between_ngrams(
    c: Candidate,
    attrib: str = "words",
    n_min: int = 1,
    n_max: int = 1,
    lower: bool = True,
) -> Iterator[str]:
    """Return the ngrams *between* two unary Mentions of a binary-Mention Candidate.

    Get the ngrams *between* two unary Mentions of a binary-Mention Candidate,
    where both share the same sentence Context.

    :param c: The binary-Mention Candidate to evaluate.
    :param attrib: The token attribute type (e.g. words, lemmas, poses)
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If 'True', all ngrams will be returned in lower case
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


def get_left_ngrams(
    mention: Union[Candidate, Mention, TemporarySpanMention],
    window: int = 3,
    attrib: str = "words",
    n_min: int = 1,
    n_max: int = 1,
    lower: bool = True,
) -> Iterator[str]:
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


def get_right_ngrams(
    mention: Union[Candidate, Mention, TemporarySpanMention],
    window: int = 3,
    attrib: str = "words",
    n_min: int = 1,
    n_max: int = 1,
    lower: bool = True,
) -> Iterator[str]:
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
    spans = _to_spans(mention)
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
    spans = _to_spans(mention)
    for span in spans:
        for ngram in chain.from_iterable(
            [
                tokens_to_ngrams(
                    getattr(sentence, attrib), n_min=n_min, n_max=n_max, lower=lower
                )
                for sentence in span.sentence.document.sentences
                if abs(sentence.position - span.sentence.position) <= d
                and sentence != span.sentence
            ]
        ):
            yield ngram
