############################
# Structural modality helpers
############################

from builtins import str

import numpy as np
from lxml import etree
from lxml.html import fromstring

from fonduer.candidates.models import TemporarySpan


def get_tag(span):
    """Return the HTML tag of the Span.

    If a candidate is passed in, only the tag of its first Span is returned.

    These may be tags such as 'p', 'h2', 'table', 'div', etc.
    :param span: The Span to evaluate
    :rtype: string
    """
    span = span if isinstance(span, TemporarySpan) else span[0]
    return str(span.sentence.html_tag)


def get_attributes(span):
    """Return the HTML attributes of the Span.

    If a candidate is passed in, only the tag of its first Span is returned.

    A sample outout of this function on a Span in a paragraph tag is
    [u'style=padding-top: 8pt;padding-left: 20pt;text-indent: 0pt;text-align: left;']

    :param span: The Span to evaluate
    :rtype: list of strings representing HTML attributes
    """
    span = span if isinstance(span, TemporarySpan) else span[0]
    return span.sentence.html_attrs


# TODO: Too slow
def _get_node(sentence):
    return (
        etree.ElementTree(fromstring(sentence.document.text)).xpath(sentence.xpath)
    )[0]


def get_parent_tag(span):
    """Return the HTML tag of the Span's parent.

    These may be tags such as 'p', 'h2', 'table', 'div', etc.
    If a candidate is passed in, only the tag of its first Span is returned.

    :param span: The Span to evaluate
    :rtype: string
    """
    span = span if isinstance(span, TemporarySpan) else span[0]
    i = _get_node(span.sentence)
    return str(i.getparent().tag) if i.getparent() is not None else None


def get_prev_sibling_tags(span):
    """Return the HTML tag of the Span's previous siblings.

    Previous siblings are Spans which are at the same level in the HTML tree as
    the given span, but are declared before the given span.
    If a candidate is passed in, only the previous siblings of its first Span
    are considered in the calculation.

    :param span: The Span to evaluate
    :rtype: list of strings
    """
    span = span if isinstance(span, TemporarySpan) else span[0]
    prev_sibling_tags = []
    i = _get_node(span.sentence)
    while i.getprevious() is not None:
        prev_sibling_tags.insert(0, str(i.getprevious().tag))
        i = i.getprevious()
    return prev_sibling_tags


def get_next_sibling_tags(span):
    """Return the HTML tag of the Span's next siblings.

    Next siblings are Spans which are at the same level in the HTML tree as
    the given span, but are declared after the given span.
    If a candidate is passed in, only the next siblings of its last Span
    are considered in the calculation.

    :param span: The Span to evaluate
    :rtype: list of strings
    """
    span = span if isinstance(span, TemporarySpan) else span[-1]
    next_sibling_tags = []
    i = _get_node(span.sentence)
    while i.getnext() is not None:
        next_sibling_tags.append(str(i.getnext().tag))
        i = i.getnext()
    return next_sibling_tags


def get_ancestor_class_names(span):
    """Return the HTML classes of the Span's ancestors.

    If a candidate is passed in, only the ancestors of its first Span are returned.

    :param span: The Span to evaluate
    :rtype: list of strings
    """
    span = span if isinstance(span, TemporarySpan) else span[0]
    class_names = []
    i = _get_node(span.sentence)
    while i is not None:
        class_names.insert(0, str(i.get("class")))
        i = i.getparent()
    return class_names


def get_ancestor_tag_names(span):
    """Return the HTML tag of the Span's ancestors.

    For example, ['html', 'body', 'p'].
    If a candidate is passed in, only the ancestors of its first Span are returned.

    :param span: The Span to evaluate
    :rtype: list of strings
    """
    span = span if isinstance(span, TemporarySpan) else span[0]
    tag_names = []
    i = _get_node(span.sentence)
    while i is not None:
        tag_names.insert(0, str(i.tag))
        i = i.getparent()
    return tag_names


def get_ancestor_id_names(span):
    """Return the HTML id's of the Span's ancestors.

    If a candidate is passed in, only the ancestors of its first Span are returned.

    :param span: The Span to evaluate
    :rtype: list of strings
    """
    span = span if isinstance(span, TemporarySpan) else span[0]
    id_names = []
    i = _get_node(span.sentence)
    while i is not None:
        id_names.insert(0, str(i.get("id")))
        i = i.getparent()
    return id_names


def common_ancestor(c):
    """Return the common path to the root that is shared between a binary-Span Candidate.

    In particular, this is the common path of HTML tags.

    :param c: The binary-Span Candidate to evaluate
    :rtype: list of strings
    """
    ancestor1 = np.array(c[0].sentence.xpath.split("/"))
    ancestor2 = np.array(c[1].sentence.xpath.split("/"))
    min_len = min(ancestor1.size, ancestor2.size)
    return list(ancestor1[: np.argmin(ancestor1[:min_len] == ancestor2[:min_len])])


def lowest_common_ancestor_depth(c):
    """Return the minimum distance between a binary-Span Candidate to their lowest common ancestor.

    For example, if the tree looked like this::

        html
        ├──<div> span 1 </div>
        ├──table
        │    ├──tr
        │    │  └──<th> span 2 </th>

    we return 1, the distance from span 1 to the html root. Smaller values indicate
    that two Spans are close structurally, while larger values indicate that two
    Spans are spread far apart structurally in the document.

    :param c: The binary-Span Candidate to evaluate
    :rtype: integer
    """
    ancestor1 = np.array(c[0].sentence.xpath.split("/"))
    ancestor2 = np.array(c[1].sentence.xpath.split("/"))
    min_len = min(ancestor1.size, ancestor2.size)
    return min_len - np.argmin(ancestor1[:min_len] == ancestor2[:min_len])
