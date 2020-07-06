"""Fonduer tree structs."""
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

from lxml import etree as et
from lxml.etree import _Element

from fonduer.parser.models import Sentence
from fonduer.utils.utils import get_as_dict


class XMLTree:
    """A generic tree representation which takes XML as input.

    Includes subroutines for conversion to JSON & for visualization based on js
    form
    """

    def __init__(self, xml_root: _Element, words: Optional[List[str]] = None) -> None:
        """Call subroutines to generate JSON form of XML input."""
        self.root = xml_root
        self.words = words

        # create a unique id for e.g. canvas id in notebook
        self.id = str(abs(hash(self.to_str())))

    def _to_json(self, root: _Element) -> Dict:
        children: List[Dict] = []
        for c in root:
            children.append(self._to_json(c))
        js = {"attrib": dict(root.attrib), "children": children}
        return js

    def to_json(self) -> Dict:
        """Convert to json."""
        return self._to_json(self.root)

    def to_str(self) -> bytes:
        """Convert to string."""
        return et.tostring(self.root)


@lru_cache(maxsize=1024)
def corenlp_to_xmltree(obj: Union[Dict, Sentence], prune_root: bool = True) -> XMLTree:
    """Convert CoreNLP attributes into an XMLTree.

    Transform an object with CoreNLP dep_path and dep_parent attributes into
    an XMLTree. Will include elements of any array having the same dimension
    as dep_* as node attributes. Also adds special word_idx attribute
    corresponding to original sequence order in sentence.
    """
    # Convert input object to dictionary
    s: Dict = get_as_dict(obj)

    # Use the dep_parents array as a guide: ensure it is present and a list of ints
    if not ("dep_parents" in s and isinstance(s["dep_parents"], list)):
        raise ValueError(
            "Input CoreNLP object must have a 'dep_parents' attribute which is a list"
        )
    try:
        dep_parents = list(map(int, s["dep_parents"]))
    except Exception:
        raise ValueError("'dep_parents' attribute must be a list of ints")

    # Also ensure that we are using CoreNLP-native indexing
    # (root=0, 1-base word indexes)!
    b = min(dep_parents)
    if b != 0:
        dep_parents = list(map(lambda j: j - b, dep_parents))

    # Parse recursively
    root = corenlp_to_xmltree_sub(s, dep_parents, 0)

    # Often the return tree will have several roots, where one is the actual
    # root and the rest are just singletons not included in the dep tree
    # parse...
    # We optionally remove these singletons and then collapse the root if only
    # one child left.
    if prune_root:
        for c in root:
            if len(c) == 0:
                root.remove(c)
        if len(root) == 1:
            root = root.findall("./*")[0]
    return XMLTree(root, words=s["words"])


def scrub(s: str) -> str:
    """Scrub the string.

    :param s: The input string.
    :return: The scrubbed string.
    """
    return "".join(c for c in s if ord(c) < 128)


def corenlp_to_xmltree_sub(
    s: Dict[str, Any], dep_parents: List[int], rid: int = 0
) -> _Element:
    """Construct XMLTree with CoreNLP information.

    :param s: Input object.
    :param dep_parents: Dependency parents.
    :param rid: Root id, defaults to 0
    :return: The constructed XMLTree.
    """
    i = rid - 1
    attrib = {}
    N = len(list(dep_parents))

    # Add all attributes that have the same shape as dep_parents
    if i >= 0:
        for k, v in list(
            filter(lambda t: isinstance(t[1], list) and len(t[1]) == N, s.items())
        ):
            if v[i] is not None:
                attrib[singular(k)] = (
                    scrub(v[i]).encode("ascii", "ignore")
                    if hasattr(v[i], "encode")
                    else str(v[i])
                )

        # Add word_idx if not present
        if "word_idx" not in attrib:
            attrib["word_idx"] = str(i)

    # Build tree recursively
    root = et.Element("node", attrib=attrib)
    for i, d in enumerate(dep_parents):
        if d == rid:
            root.append(corenlp_to_xmltree_sub(s, dep_parents, i + 1))
    return root


def singular(s: str) -> str:
    """Get singular form of word s (crudely).

    :param s: The input string.
    :return: The singular form of the string.
    """
    return re.sub(r"e?s$", "", s, flags=re.I)
