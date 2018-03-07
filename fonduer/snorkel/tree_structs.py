import json
import os
import re
import six
import lxml.etree as et
from fonduer.snorkel.utils import corenlp_cleaner

APP_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Load IPython display functionality libs if possible i.e. if in IPython
try:
    from IPython.core.display import display_html, HTML, display_javascript, Javascript
except:
    pass


class XMLTree:
    """
    A generic tree representation which takes XML as input
    Includes subroutines for conversion to JSON & for visualization based on js form
    """

    def __init__(self, xml_root, words=None):
        """Calls subroutines to generate JSON form of XML input"""
        self.root = xml_root
        self.words = words

        # create a unique id for e.g. canvas id in notebook
        self.id = str(abs(hash(self.to_str())))

    def _to_json(self, root):
        js = {'attrib': dict(root.attrib), 'children': []}
        for i, c in enumerate(root):
            js['children'].append(self._to_json(c))
        return js

    def to_json(self):
        return self._to_json(self.root)

    def to_str(self):
        return et.tostring(self.root)

    def render_tree(self, highlight=[]):
        """
        Renders d3 visualization of the d3 tree, for IPython notebook display
        Depends on html/js files in vis/ directory, which is assumed to be in same dir...
        """
        # HTML
        WORD = '<span class="word-' + self.id + '-%s">%s</span>'
        words = ' '.join(WORD % (i, w) for i, w in enumerate(
            corenlp_cleaner(self.words))) if self.words else ''
        html = open('%s/vis/tree-chart.html' % APP_HOME).read() % (self.id,
                                                                   self.id,
                                                                   words)
        display_html(HTML(data=html))

        # JS
        JS_LIBS = ["https://d3js.org/d3.v3.min.js"]
        js = open('%s/vis/tree-chart.js' % APP_HOME).read() % (
            self.id, json.dumps(self.to_json()), str(highlight))
        display_javascript(Javascript(data=js, lib=JS_LIBS))


def corenlp_to_xmltree(s, prune_root=True):
    """
    Transforms an object with CoreNLP dep_path and dep_parent attributes into an XMLTree
    Will include elements of any array having the same dimensiion as dep_* as node attributes
    Also adds special word_idx attribute corresponding to original sequence order in sentence
    """
    # Convert input object to dictionary
    if isinstance(s, dict) == False:
        try:
            s = s.__dict__ if hasattr(s, '__dict__') else dict(s)
        except:
            raise ValueError("Cannot convert input object to dict")

    # Use the dep_parents array as a guide: ensure it is present and a list of ints
    if not ('dep_parents' in s and isinstance(s['dep_parents'], list)):
        raise ValueError(
            "Input CoreNLP object must have a 'dep_parents' attribute which is a list"
        )
    try:
        dep_parents = list(map(int, s['dep_parents']))
    except:
        raise ValueError("'dep_parents' attribute must be a list of ints")

    # Also ensure that we are using CoreNLP-native indexing (root=0, 1-base word indexes)!
    b = min(dep_parents)
    if b != 0:
        dep_parents = list(map(lambda j: j - b, dep_parents))

    # Parse recursively
    root = corenlp_to_xmltree_sub(s, dep_parents, 0)

    # Often the return tree will have several roots, where one is the actual root
    # And the rest are just singletons not included in the dep tree parse...
    # We optionally remove these singletons and then collapse the root if only one child left
    if prune_root:
        for c in root:
            if len(c) == 0:
                root.remove(c)
        if len(root) == 1:
            root = root.findall("./*")[0]
    return XMLTree(root, words=s['words'])


def scrub(s):
    return ''.join(c for c in s if ord(c) < 128)


def corenlp_to_xmltree_sub(s, dep_parents, rid=0):
    i = rid - 1
    attrib = {}
    N = len(list(dep_parents))

    # Add all attributes that have the same shape as dep_parents
    if i >= 0:
        for k, v in list(
                filter(lambda t: isinstance(t[1], list) and len(t[1]) == N,
                       six.iteritems(s))):
            if v[i] is not None:
                attrib[singular(k)] = scrub(v[i]).encode(
                    'ascii', 'ignore') if hasattr(v[i], 'encode') else str(
                        v[i])

        # Add word_idx if not present
        if 'word_idx' not in attrib:
            attrib['word_idx'] = str(i)

    # Build tree recursively
    root = et.Element('node', attrib=attrib)
    for i, d in enumerate(dep_parents):
        if d == rid:
            root.append(corenlp_to_xmltree_sub(s, dep_parents, i + 1))
    return root


def singular(s):
    """Get singular form of word s (crudely)"""
    return re.sub(r'e?s$', '', s, flags=re.I)
