###############################
# Structural modality utilities
###############################

import functools
from builtins import str

import numpy as np
from lxml import etree
from lxml.html import fromstring

from fonduer.utils.data_model_utils.utils import _to_span


def get_tag(mention):
    """Return the HTML tag of the Mention.

    If a candidate is passed in, only the tag of its first Mention is returned.

    These may be tags such as 'p', 'h2', 'table', 'div', etc.
    :param mention: The Mention to evaluate
    :rtype: string
    """
    span = _to_span(mention)
    return str(span.sentence.html_tag)


def get_attributes(mention):
    """Return the HTML attributes of the Mention.

    If a candidate is passed in, only the tag of its first Mention is returned.

    A sample outout of this function on a Mention in a paragraph tag is
    [u'style=padding-top: 8pt;padding-left: 20pt;text-indent: 0pt;text-align: left;']

    :param mention: The Mention to evaluate
    :rtype: list of strings representing HTML attributes
    """
    span = _to_span(mention)
    return span.sentence.html_attrs


@functools.lru_cache(maxsize=16)
def _get_etree_for_text(text):
    return etree.ElementTree(fromstring(text))


def _get_node(sentence):
    # Using caching to speed up retrieve process
    doc_etree = _get_etree_for_text(sentence.document.text)
    return doc_etree.xpath(sentence.xpath)[0]


def get_parent_tag(mention):
    """Return the HTML tag of the Mention's parent.

    These may be tags such as 'p', 'h2', 'table', 'div', etc.
    If a candidate is passed in, only the tag of its first Mention is returned.

    :param mention: The Mention to evaluate
    :rtype: string
    """
    span = _to_span(mention)
    i = _get_node(span.sentence)
    return str(i.getparent().tag) if i.getparent() is not None else None


def get_prev_sibling_tags(mention):
    """Return the HTML tag of the Mention's previous siblings.

    Previous siblings are Mentions which are at the same level in the HTML tree
    as the given mention, but are declared before the given mention. If a
    candidate is passed in, only the previous siblings of its first Mention are
    considered in the calculation.

    :param mention: The Mention to evaluate
    :rtype: list of strings
    """
    span = _to_span(mention)
    prev_sibling_tags = []
    i = _get_node(span.sentence)
    while i.getprevious() is not None:
        prev_sibling_tags.insert(0, str(i.getprevious().tag))
        i = i.getprevious()
    return prev_sibling_tags


def get_next_sibling_tags(mention):
    """Return the HTML tag of the Mention's next siblings.

    Next siblings are Mentions which are at the same level in the HTML tree as
    the given mention, but are declared after the given mention.
    If a candidate is passed in, only the next siblings of its last Mention
    are considered in the calculation.

    :param mention: The Mention to evaluate
    :rtype: list of strings
    """
    span = _to_span(mention)
    next_sibling_tags = []
    i = _get_node(span.sentence)
    while i.getnext() is not None:
        next_sibling_tags.append(str(i.getnext().tag))
        i = i.getnext()
    return next_sibling_tags


def get_ancestor_class_names(mention):
    """Return the HTML classes of the Mention's ancestors.

    If a candidate is passed in, only the ancestors of its first Mention are
    returned.

    :param mention: The Mention to evaluate
    :rtype: list of strings
    """
    span = _to_span(mention)
    class_names = []
    i = _get_node(span.sentence)
    while i is not None:
        class_names.insert(0, str(i.get("class")))
        i = i.getparent()
    return class_names


def get_ancestor_tag_names(mention):
    """Return the HTML tag of the Mention's ancestors.

    For example, ['html', 'body', 'p'].
    If a candidate is passed in, only the ancestors of its first Mention are returned.

    :param mention: The Mention to evaluate
    :rtype: list of strings
    """
    span = _to_span(mention)
    tag_names = []
    i = _get_node(span.sentence)
    while i is not None:
        tag_names.insert(0, str(i.tag))
        i = i.getparent()
    return tag_names


def get_ancestor_id_names(mention):
    """Return the HTML id's of the Mention's ancestors.

    If a candidate is passed in, only the ancestors of its first Mention are
    returned.

    :param mention: The Mention to evaluate
    :rtype: list of strings
    """
    span = _to_span(mention)
    id_names = []
    i = _get_node(span.sentence)
    while i is not None:
        id_names.insert(0, str(i.get("id")))
        i = i.getparent()
    return id_names


def common_ancestor(c):
    """Return the path to the root that is shared between a binary-Mention Candidate.

    In particular, this is the common path of HTML tags.

    :param c: The binary-Mention Candidate to evaluate
    :rtype: list of strings
    """
    span1 = _to_span(c[0])
    span2 = _to_span(c[1])
    ancestor1 = np.array(span1.sentence.xpath.split("/"))
    ancestor2 = np.array(span2.sentence.xpath.split("/"))
    min_len = min(ancestor1.size, ancestor2.size)
    return list(ancestor1[: np.argmin(ancestor1[:min_len] == ancestor2[:min_len])])


def lowest_common_ancestor_depth(c):
    """Return the minimum distance between a binary-Mention Candidate to their
    lowest common ancestor.

    For example, if the tree looked like this::

        html
        ├──<div> Mention 1 </div>
        ├──table
        │    ├──tr
        │    │  └──<th> Mention 2 </th>

    we return 1, the distance from Mention 1 to the html root. Smaller values
    indicate that two Mentions are close structurally, while larger values
    indicate that two Mentions are spread far apart structurally in the
    document.

    :param c: The binary-Mention Candidate to evaluate
    :rtype: integer
    """
    span1 = _to_span(c[0])
    span2 = _to_span(c[1])
    ancestor1 = np.array(span1.sentence.xpath.split("/"))
    ancestor2 = np.array(span2.sentence.xpath.split("/"))
    min_len = min(ancestor1.size, ancestor2.size)
    return min_len - np.argmin(ancestor1[:min_len] == ancestor2[:min_len])
