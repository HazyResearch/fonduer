"""Fonduer visual modality utilities."""
from builtins import range
from collections import defaultdict
from functools import lru_cache
from typing import Any, DefaultDict, Iterator, List, Set, Union

from fonduer.candidates.mentions import Ngrams
from fonduer.candidates.models import Candidate, Mention
from fonduer.candidates.models.span_mention import TemporarySpanMention
from fonduer.parser.models import Document, Sentence
from fonduer.utils.data_model_utils.utils import _to_span, _to_spans
from fonduer.utils.utils import tokens_to_ngrams
from fonduer.utils.utils_visual import (
    bbox_horz_aligned,
    bbox_vert_aligned,
    bbox_vert_aligned_center,
    bbox_vert_aligned_left,
    bbox_vert_aligned_right,
)


@lru_cache(maxsize=1024)
def get_page(mention: Union[Candidate, Mention, TemporarySpanMention]) -> int:
    """Return the page number of the given mention.

    If a candidate is passed in, this returns the page of its first Mention.

    :param mention: The Mention to get the page number of.
    """
    span = _to_span(mention)
    return span.get_attrib_tokens("page")[0]


@lru_cache(maxsize=1024)
def is_horz_aligned(c: Candidate) -> bool:
    """Return True if all the components of c are horizontally aligned.

    Horizontal alignment means that the bounding boxes of each Mention of c
    shares a similar y-axis value in the visual rendering of the document.

    :param c: The candidate to evaluate
    """
    return all(
        [
            _to_span(c[i]).sentence.is_visual()
            and bbox_horz_aligned(_to_span(c[i]).get_bbox(), _to_span(c[0]).get_bbox())
            for i in range(len(c))
        ]
    )


@lru_cache(maxsize=1024)
def is_vert_aligned(c: Candidate) -> bool:
    """Return true if all the components of c are vertically aligned.

    Vertical alignment means that the bounding boxes of each Mention of c
    shares a similar x-axis value in the visual rendering of the document.

    :param c: The candidate to evaluate
    """
    return all(
        [
            _to_span(c[i]).sentence.is_visual()
            and bbox_vert_aligned(_to_span(c[i]).get_bbox(), _to_span(c[0]).get_bbox())
            for i in range(len(c))
        ]
    )


@lru_cache(maxsize=1024)
def is_vert_aligned_left(c: Candidate) -> bool:
    """Return true if all components are vertically aligned on their left border.

    Vertical alignment means that the bounding boxes of each Mention of c
    shares a similar x-axis value in the visual rendering of the document. In
    this function the similarity of the x-axis value is based on the left
    border of their bounding boxes.

    :param c: The candidate to evaluate
    """
    return all(
        [
            _to_span(c[i]).sentence.is_visual()
            and bbox_vert_aligned_left(
                _to_span(c[i]).get_bbox(), _to_span(c[0]).get_bbox()
            )
            for i in range(len(c))
        ]
    )


@lru_cache(maxsize=1024)
def is_vert_aligned_right(c: Candidate) -> bool:
    """Return true if all components vertically aligned on their right border.

    Vertical alignment means that the bounding boxes of each Mention of c
    shares a similar x-axis value in the visual rendering of the document. In
    this function the similarity of the x-axis value is based on the right
    border of their bounding boxes.

    :param c: The candidate to evaluate
    """
    return all(
        [
            _to_span(c[i]).sentence.is_visual()
            and bbox_vert_aligned_right(
                _to_span(c[i]).get_bbox(), _to_span(c[0]).get_bbox()
            )
            for i in range(len(c))
        ]
    )


@lru_cache(maxsize=1024)
def is_vert_aligned_center(c: Candidate) -> bool:
    """Return true if all the components are vertically aligned on their center.

    Vertical alignment means that the bounding boxes of each Mention of c
    shares a similar x-axis value in the visual rendering of the document. In
    this function the similarity of the x-axis value is based on the center of
    their bounding boxes.

    :param c: The candidate to evaluate
    """
    return all(
        [
            _to_span(c[i]).sentence.is_visual()
            and bbox_vert_aligned_center(
                _to_span(c[i]).get_bbox(), _to_span(c[0]).get_bbox()
            )
            for i in range(len(c))
        ]
    )


@lru_cache(maxsize=1024)
def same_page(c: Candidate) -> bool:
    """Return true if all the components of c are on the same page of the document.

    Page numbers are based on the PDF rendering of the document. If a PDF file is
    provided, it is used. Otherwise, if only a HTML/XML document is provided, a
    PDF is created and then used to determine the page number of a Mention.

    :param c: The candidate to evaluate
    """
    return all(
        [
            _to_span(c[i]).sentence.is_visual()
            and _to_span(c[i]).get_bbox().page == _to_span(c[0]).get_bbox().page
            for i in range(len(c))
        ]
    )


def get_horz_ngrams(
    mention: Union[Candidate, Mention, TemporarySpanMention],
    attrib: str = "words",
    n_min: int = 1,
    n_max: int = 1,
    lower: bool = True,
    from_sentence: bool = True,
) -> Iterator[str]:
    """Return all ngrams which are visually horizontally aligned with the Mention.

    Note that if a candidate is passed in, all of its Mentions will be searched.

    :param mention: The Mention to evaluate
    :param attrib: The token attribute type (e.g. words, lemmas, pos_tags).
        This option is valid only when ``from_sentence==True``.
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    :param from_sentence: If True, return ngrams of any ``Sentence`` that is
        horizontally aligned (in the same page) with the mention's ``Sentence``.
        If False, return ngrams that are horizontally aligned with the mention
        no matter which ``Sentence`` they are from.
    :return: a *generator* of ngrams
    """
    spans = _to_spans(mention)
    for span in spans:
        for ngram in _get_direction_ngrams(
            "horz", span, attrib, n_min, n_max, lower, from_sentence
        ):
            yield ngram


def get_vert_ngrams(
    mention: Union[Candidate, Mention, TemporarySpanMention],
    attrib: str = "words",
    n_min: int = 1,
    n_max: int = 1,
    lower: bool = True,
    from_sentence: bool = True,
) -> Iterator[str]:
    """Return all ngrams which are visually vertically aligned with the Mention.

    Note that if a candidate is passed in, all of its Mentions will be searched.

    :param mention: The Mention to evaluate
    :param attrib: The token attribute type (e.g. words, lemmas, pos_tags).
        This option is valid only when ``from_sentence==True``.
    :param n_min: The minimum n of the ngrams that should be returned
    :param n_max: The maximum n of the ngrams that should be returned
    :param lower: If True, all ngrams will be returned in lower case
    :param from_sentence: If True, return ngrams of any ``Sentence`` that is
        vertically aligned (in the same page) with the mention's ``Sentence``.
        If False, return ngrams that are vertically aligned with the mention
        no matter which ``Sentence`` they are from.
    :return: a *generator* of ngrams
    """
    spans = _to_spans(mention)
    for span in spans:
        for ngram in _get_direction_ngrams(
            "vert", span, attrib, n_min, n_max, lower, from_sentence
        ):
            yield ngram


def _get_direction_ngrams(
    direction: str,
    c: Union[Candidate, Mention, TemporarySpanMention],
    attrib: str,
    n_min: int,
    n_max: int,
    lower: bool,
    from_sentence: bool,
) -> Iterator[str]:
    bbox_direction_aligned = (
        bbox_vert_aligned if direction == "vert" else bbox_horz_aligned
    )
    ngrams_space = Ngrams(n_max=n_max, split_tokens=[])
    f = (lambda w: w.lower()) if lower else (lambda w: w)
    spans = _to_spans(c)
    for span in spans:
        if not span.sentence.is_visual():
            continue
        for sentence in span.sentence.document.sentences:
            # Skip if not in the same page.
            if span.sentence.get_bbox().page != sentence.get_bbox().page:
                continue
            if from_sentence:
                if (
                    bbox_direction_aligned(sentence.get_bbox(), span.get_bbox())
                    and sentence is not span.sentence  # not from its Sentence
                ):
                    for ngram in tokens_to_ngrams(
                        getattr(sentence, attrib), n_min=n_min, n_max=n_max, lower=lower
                    ):
                        yield ngram
            else:
                for ts in ngrams_space.apply(sentence):
                    if (  # True if visually aligned AND not from itself.
                        bbox_direction_aligned(ts.get_bbox(), span.get_bbox())
                        and ts not in span
                        and span not in ts
                    ):
                        yield f(ts.get_span())


def get_vert_ngrams_left(c):  # type: ignore
    """Not implemented."""
    # TODO
    return


def get_vert_ngrams_right(c):  # type: ignore
    """Not implemented."""
    # TODO
    return


def get_vert_ngrams_center(c):  # type: ignore
    """Not implemented."""
    # TODO
    return


def get_visual_header_ngrams(c, axis=None):  # type: ignore
    """Not implemented."""
    # TODO
    return


def get_visual_distance(c, axis=None):  # type: ignore
    """Not implemented."""
    # TODO
    return


# Default dimensions for 8.5" x 11"
DEFAULT_WIDTH = 612
DEFAULT_HEIGHT = 792


def get_page_vert_percentile(
    mention: Union[Candidate, Mention, TemporarySpanMention],
    page_width: int = DEFAULT_WIDTH,
    page_height: int = DEFAULT_HEIGHT,
) -> float:
    """Return which percentile from the TOP in the page the Mention is located in.

    Percentile is calculated where the top of the page is 0.0, and the bottom
    of the page is 1.0. For example, a Mention in at the top 1/4 of the page
    will have a percentile of 0.25.

    Page width and height are based on pt values::

        Letter      612x792
        Tabloid     792x1224
        Ledger      1224x792
        Legal       612x1008
        Statement   396x612
        Executive   540x720
        A0          2384x3371
        A1          1685x2384
        A2          1190x1684
        A3          842x1190
        A4          595x842
        A4Small     595x842
        A5          420x595
        B4          729x1032
        B5          516x729
        Folio       612x936
        Quarto      610x780
        10x14       720x1008

    and should match the source documents. Letter size is used by default.

    Note that if a candidate is passed in, only the vertical percentil of its
    first Mention is returned.

    :param mention: The Mention to evaluate
    :param page_width: The width of the page. Default to Letter paper width.
    :param page_height: The heigh of the page. Default to Letter paper height.
    """
    span = _to_span(mention)
    return span.get_bbox().top / page_height


def get_page_horz_percentile(
    mention: Union[Candidate, Mention, TemporarySpanMention],
    page_width: int = DEFAULT_WIDTH,
    page_height: int = DEFAULT_HEIGHT,
) -> float:
    """Return which percentile from the LEFT in the page the Mention is located in.

    Percentile is calculated where the left of the page is 0.0, and the right
    of the page is 1.0.

    Page width and height are based on pt values::

        Letter      612x792
        Tabloid     792x1224
        Ledger      1224x792
        Legal       612x1008
        Statement   396x612
        Executive   540x720
        A0          2384x3371
        A1          1685x2384
        A2          1190x1684
        A3          842x1190
        A4          595x842
        A4Small     595x842
        A5          420x595
        B4          729x1032
        B5          516x729
        Folio       612x936
        Quarto      610x780
        10x14       720x1008

    and should match the source documents. Letter size is used by default.

    Note that if a candidate is passed in, only the vertical percentile of its
    first Mention is returned.

    :param mention: The Mention to evaluate
    :param page_width: The width of the page. Default to Letter paper width.
    :param page_height: The heigh of the page. Default to Letter paper height.
    """
    span = _to_span(mention)
    return span.get_bbox().left / page_width


def _assign_alignment_features(sentences_by_key: defaultdict, align_type: str) -> None:
    for key, sentences in sentences_by_key.items():
        if len(sentences) == 1:
            continue
        context_lemmas: Set[str] = set()
        for p in sentences:
            p._aligned_lemmas.update(context_lemmas)
            # update lemma context for upcoming sentences in the group
            if len(p.lemmas) < 7:
                new_lemmas = [lemma.lower() for lemma in p.lemmas if lemma.isalpha()]
                context_lemmas.update(new_lemmas)
                context_lemmas.update(align_type + lemma for lemma in new_lemmas)


@lru_cache(maxsize=2)
def _preprocess_visual_features(doc: Document) -> None:
    if hasattr(doc, "_visual_features"):
        return
    # cache flag
    doc._visual_features = True

    sentence_by_page: DefaultDict[str, List[Sentence]] = defaultdict(list)
    for sentence in doc.sentences:
        sentence_by_page[sentence.page[0]].append(sentence)
        sentence._aligned_lemmas = set()

    for page, sentences in sentence_by_page.items():
        # process per page alignments
        yc_aligned: DefaultDict[Any, List[Sentence]] = defaultdict(list)
        x0_aligned: DefaultDict[Any, List[Sentence]] = defaultdict(list)
        xc_aligned: DefaultDict[Any, List[Sentence]] = defaultdict(list)
        x1_aligned: DefaultDict[Any, List[Sentence]] = defaultdict(list)
        for sentence in sentences:
            sentence.bbox = sentence.get_bbox()
            sentence.yc = (sentence.bbox.top + sentence.bbox.bottom) / 2
            sentence.x0 = sentence.bbox.left
            sentence.x1 = sentence.bbox.right
            sentence.xc = (sentence.x0 + sentence.x1) / 2
            # index current sentence by different alignment keys
            yc_aligned[sentence.yc].append(sentence)
            x0_aligned[sentence.x0].append(sentence)
            x1_aligned[sentence.x1].append(sentence)
            xc_aligned[sentence.xc].append(sentence)
        for l in yc_aligned.values():
            l.sort(key=lambda p: p.xc)
        for l in x0_aligned.values():
            l.sort(key=lambda p: p.yc)
        for l in x1_aligned.values():
            l.sort(key=lambda p: p.yc)
        for l in xc_aligned.values():
            l.sort(key=lambda p: p.yc)
        _assign_alignment_features(yc_aligned, "Y_")
        _assign_alignment_features(x0_aligned, "LEFT_")
        _assign_alignment_features(x1_aligned, "RIGHT_")
        _assign_alignment_features(xc_aligned, "CENTER_")


def get_visual_aligned_lemmas(
    mention: Union[Candidate, Mention, TemporarySpanMention]
) -> Iterator[str]:
    """Return a generator of the lemmas aligned visually with the Mention.

    Note that if a candidate is passed in, all of its Mentions will be searched.

    :param mention: The Mention to evaluate.
    """
    spans = _to_spans(mention)
    for span in spans:
        sentence = span.sentence
        doc = sentence.document
        # cache features for the entire document
        _preprocess_visual_features(doc)

        for aligned_lemma in sentence._aligned_lemmas:
            yield aligned_lemma


def get_aligned_lemmas(
    mention: Union[Candidate, Mention, TemporarySpanMention]
) -> Set[str]:
    """Return a set of the lemmas aligned visually with the Mention.

    Note that if a candidate is passed in, all of its Mentions will be searched.

    :param mention: The Mention to evaluate.
    """
    return set(get_visual_aligned_lemmas(mention))
