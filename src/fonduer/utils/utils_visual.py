"""Fonduer visual utils."""
import warnings
from typing import NamedTuple


class Bbox(NamedTuple):
    """Bounding box."""

    page: int
    top: int
    bottom: int
    left: int
    right: int


def bbox_from_span(span) -> Bbox:  # type: ignore
    """Get bounding box from span.

    :param span: The input span.
    :return: The bounding box of the span.
    """
    warnings.warn(
        "bbox_from_span(span) is deprecated. Use span.get_bbox() instead.",
        DeprecationWarning,
    )
    from fonduer.candidates.models.span_mention import TemporarySpanMention  # noqa

    if isinstance(span, TemporarySpanMention) and span.sentence.is_visual():
        return Bbox(
            span.get_attrib_tokens("page")[0],
            min(span.get_attrib_tokens("top")),
            max(span.get_attrib_tokens("bottom")),
            min(span.get_attrib_tokens("left")),
            max(span.get_attrib_tokens("right")),
        )
    else:
        return None


def bbox_from_sentence(sentence) -> Bbox:  # type: ignore
    """Get bounding box from sentence.

    :param sentence: The input sentence.
    :return: The bounding box of the sentence.
    """
    warnings.warn(
        "bbox_from_sentence(sentence) is deprecated. Use sentence.get_bbox() instead.",
        DeprecationWarning,
    )
    from fonduer.parser.models import Sentence  # noqa

    # TODO: this may have issues where a sentence is linked to words on different pages
    if isinstance(sentence, Sentence) and sentence.is_visual():
        return Bbox(
            sentence.page[0],
            min(sentence.top),
            max(sentence.bottom),
            min(sentence.left),
            max(sentence.right),
        )
    else:
        return None


def bbox_horz_aligned(box1: Bbox, box2: Bbox) -> bool:
    """Check two bounding boxes are horizontally aligned.

    Return true if the vertical center point of either span is within the
    vertical range of the other
    """
    if not (box1 and box2):
        return False
    # NEW: any overlap counts
    #    return box1.top <= box2.bottom and box2.top <= box1.bottom
    box1_top = box1.top + 1.5
    box2_top = box2.top + 1.5
    box1_bottom = box1.bottom - 1.5
    box2_bottom = box2.bottom - 1.5
    return not (box1_top > box2_bottom or box2_top > box1_bottom)


#    return not (box1.top >= box2.bottom or box2.top >= box1.bottom)
#    center1 = (box1.bottom + box1.top) / 2.0
#    center2 = (box2.bottom + box2.top) / 2.0
#    return ((center1 >= box2.top and center1 <= box2.bottom) or
#            (center2 >= box1.top and center2 <= box1.bottom))


def bbox_vert_aligned(box1: Bbox, box2: Bbox) -> bool:
    """Check two bounding boxes are vertical aligned.

    Return true if the horizontal center point of either span is within the
    horizontal range of the other
    """
    if not (box1 and box2):
        return False
    # NEW: any overlap counts
    #    return box1.left <= box2.right and box2.left <= box1.right
    box1_left = box1.left + 1.5
    box2_left = box2.left + 1.5
    box1_right = box1.right - 1.5
    box2_right = box2.right - 1.5
    return not (box1_left > box2_right or box2_left > box1_right)
    # center1 = (box1.right + box1.left) / 2.0
    # center2 = (box2.right + box2.left) / 2.0
    # return ((center1 >= box2.left and center1 <= box2.right) or
    #         (center2 >= box1.left and center2 <= box1.right))


def bbox_vert_aligned_left(box1: Bbox, box2: Bbox) -> bool:
    """Check two boxes' left boundaries are with 2pts.

    Return true if the left boundary of both boxes is within 2 pts.
    """
    if not (box1 and box2):
        return False
    return abs(box1.left - box2.left) <= 2


def bbox_vert_aligned_right(box1: Bbox, box2: Bbox) -> bool:
    """Check two boxes' right boundaries are with 2pts.

    Return true if the right boundary of both boxes is within 2 pts.
    """
    if not (box1 and box2):
        return False
    return abs(box1.right - box2.right) <= 2


def bbox_vert_aligned_center(box1: Bbox, box2: Bbox) -> bool:
    """Check two boxes' centers are with 5pts.

    Return true if the center of both boxes is within 5 pts.
    """
    if not (box1 and box2):
        return False
    return abs(((box1.right + box1.left) / 2.0) - ((box2.right + box2.left) / 2.0)) <= 5
