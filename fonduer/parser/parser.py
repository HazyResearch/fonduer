import itertools
import logging
import os
import re
from builtins import object, range, str
from collections import defaultdict

import numpy as np
from lxml import etree
from lxml.html import fromstring

from fonduer.models import (Candidate, Cell, Context, Document, Figure, Phrase,
                            Table, construct_stable_id, split_stable_id)
from fonduer.parser.spacy_parser import Spacy
from fonduer.udf import UDF, UDFRunner
from fonduer.visual import VisualLinker

logger = logging.getLogger(__name__)


class SimpleTokenizer(object):
    """
    A trivial alternative to CoreNLP which parses (tokenizes) text on
    whitespace only using the split() command.
    """

    def __init__(self, delim):
        self.delim = delim

    def parse(self, document, contents):
        i = 0
        for text in contents.split(self.delim):
            if not len(text.strip()):
                continue
            words = text.split()
            char_offsets = [0] + [int(_) for _ in np.cumsum([len(x) + 1
                                  for x in words])[:-1]]
            text = ' '.join(words)
            stable_id = construct_stable_id(document, 'phrase', i, i)
            yield {
                'text': text,
                'words': words,
                'pos_tags': [''] * len(words),
                'ner_tags': [''] * len(words),
                'lemmas': [''] * len(words),
                'dep_parents': [0] * len(words),
                'dep_labels': [''] * len(words),
                'char_offsets': char_offsets,
                'abs_char_offsets': char_offsets,
                'stable_id': stable_id
            }
            i += 1


class OmniParser(UDFRunner):
    def __init__(
            self,
            structural=True,  # structural information
            blacklist=["style"],  # ignore tag types, default: style
            flatten=['span', 'br'],  # flatten tag types, default: span, br
            flatten_delim='',
            lingual=True,  # lingual information
            strip=True,
            replacements=[(u'[\u2010\u2011\u2012\u2013\u2014\u2212\uf02d]',
                           '-')],
            tabular=True,  # tabular information
            visual=False,  # visual information
            pdf_path=None):

        self.delim = "<NB>"  # NB = New Block

        # Use spaCy as our lingual parser
        self.lingual_parser = Spacy()

        super(OmniParser, self).__init__(
            OmniParserUDF,
            structural=structural,
            blacklist=blacklist,
            flatten=flatten,
            flatten_delim=flatten_delim,
            lingual=lingual,
            strip=strip,
            replacements=replacements,
            tabular=tabular,
            visual=visual,
            pdf_path=pdf_path,
            lingual_parser=self.lingual_parser)

    def clear(self, session, **kwargs):
        session.query(Context).delete()

        # We cannot cascade up from child contexts to parent Candidates, so we delete all Candidates too
        session.query(Candidate).delete()


class OmniParserUDF(UDF):
    def __init__(
            self,
            structural,  # structural
            blacklist,
            flatten,
            flatten_delim,
            lingual,  # lingual
            strip,
            replacements,
            tabular,  # tabular
            visual,  # visual
            pdf_path,
            lingual_parser,
            **kwargs):
        """
        :param visual: boolean, if True visual features are used in the model
        :param pdf_path: directory where pdf are saved, if a pdf file is not found,
        it will be created from the html document and saved in that directory
        :param replacements: a list of (_pattern_, _replace_) tuples where _pattern_ isinstance
        a regex and _replace_ is a character string. All occurents of _pattern_ in the
        text will be replaced by _replace_.
        """
        super(OmniParserUDF, self).__init__(**kwargs)

        self.delim = "<NB>"  # NB = New Block

        # structural (html) setup
        self.structural = structural
        self.blacklist = blacklist if isinstance(blacklist,
                                                 list) else [blacklist]
        self.flatten = flatten if isinstance(flatten, list) else [flatten]
        self.flatten_delim = flatten_delim

        # lingual setup
        self.lingual = lingual
        self.strip = strip
        self.replacements = []
        for (pattern, replace) in replacements:
            self.replacements.append((re.compile(pattern, flags=re.UNICODE),
                                      replace))
        if self.lingual:
            self.lingual_parser = lingual_parser
            self.lingual_parse = self.lingual_parser.parse

        else:
            self.batch_size = int(1e6)
            self.lingual_parse = SimpleTokenizer(delim=self.delim).parse

        # tabular setup
        self.tabular = tabular

        # visual setup
        self.visual = visual
        if self.visual:
            self.pdf_path = pdf_path
            self.vizlink = VisualLinker()

    def apply(self, x, **kwargs):
        document, text = x
        if self.visual:
            if not self.pdf_path:
                logger.error("Visual parsing failed: pdf_path is required")
            for _ in self.parse_structure(document, text):
                pass
            # Add visual attributes
            filename = self.pdf_path + document.name
            create_pdf = not os.path.isfile(
                filename + '.pdf') and not os.path.isfile(
                    filename + '.PDF') and not os.path.isfile(filename)
            if create_pdf:  # PDF file does not exist
                logger.error("Visual parsing failed: pdf files are required")
            for phrase in self.vizlink.parse_visual(
                    document.name, document.phrases, self.pdf_path):
                yield phrase
        else:
            for phrase in self.parse_structure(document, text):
                yield phrase

    def _flatten(self, node):
        # if a child of this node is in self.flatten, construct a string
        # containing all text/tail results of the tree based on that child
        # and append that to the tail of the previous child or head of node
        num_children = len(node)
        for i, child in enumerate(node[::-1]):
            if child.tag in self.flatten:
                j = num_children - 1 - i  # child index walking backwards
                contents = ['']
                for descendant in child.getiterator():
                    if descendant.text and descendant.text.strip():
                        contents.append(descendant.text)
                    if descendant.tail and descendant.tail.strip():
                        contents.append(descendant.tail)
                if j == 0:
                    if node.text is None:
                        node.text = ''
                    node.text += self.flatten_delim.join(contents)
                else:
                    if node[j - 1].tail is None:
                        node[j - 1].tail = ''
                    node[j - 1].tail += self.flatten_delim.join(contents)
                node.remove(child)

    def parse_structure(self, document, text):
        self.contents = ""
        block_lengths = []
        self.parent = document

        figure_info = FigureInfo(document, parent=document)
        self.figure_idx = -1

        if self.tabular:
            table_info = TableInfo(document, parent=document)
            self.table_idx = -1
        else:
            table_info = None

        self.parsed = 0
        self.parent_idx = 0
        self.position = 0
        self.phrase_num = 0
        self.abs_phrase_offset = 0

        def parse_node(node, table_info=None, figure_info=None):
            if node.tag is etree.Comment:
                return
            if self.blacklist and node.tag in self.blacklist:
                return

            self.figure_idx = figure_info.enter_figure(node, self.figure_idx)

            if self.tabular:
                self.table_idx = table_info.enter_tabular(node, self.table_idx)

            # flattens children of node that are in the 'flatten' list
            if self.flatten:
                self._flatten(node)

            for field in ['text', 'tail']:
                text = getattr(node, field)
                if text is not None:
                    if self.strip:
                        text = text.strip()
                    if len(text):
                        for (rgx, replace) in self.replacements:
                            text = rgx.sub(replace, text)
                        self.contents += text
                        self.contents += self.delim
                        block_lengths.append(len(text) + len(self.delim))

                        for parts in self.lingual_parse(document, text):
                            (_, _, _, char_end) = split_stable_id(
                                parts['stable_id'])
                            try:
                                parts['document'] = document
                                parts['phrase_num'] = self.phrase_num
                                abs_phrase_offset_end = (
                                    self.abs_phrase_offset +
                                    parts['char_offsets'][-1] + len(
                                        parts['words'][-1]))
                                parts['stable_id'] = construct_stable_id(
                                    document, 'phrase', self.abs_phrase_offset,
                                    abs_phrase_offset_end)
                                self.abs_phrase_offset = abs_phrase_offset_end
                                if self.structural:
                                    context_node = node.getparent(
                                    ) if field == 'tail' else node
                                    parts['xpath'] = tree.getpath(context_node)
                                    parts['html_tag'] = context_node.tag
                                    parts['html_attrs'] = [
                                        '='.join(x) for x in list(
                                            context_node.attrib.items())
                                    ]

                                    # Extending html style attribute with the styles
                                    # from inline style class for the element.
                                    cur_style_index = None
                                    for index, attr in enumerate(parts['html_attrs']):
                                        if attr.find('style') >= 0:
                                            cur_style_index = index
                                            break
                                    styles = root.find('head').find('style')
                                    if styles is not None:
                                        for x in list(context_node.attrib.items()):
                                            if x[0] == 'class':
                                                exp = r'(.' + x[1] + ')([\n\s\r]*)\{(.*?)\}'
                                                r = re.compile(exp, re.DOTALL)
                                                if r.search(styles.text) is not None:
                                                    if cur_style_index is not None:
                                                        parts['html_attrs'][cur_style_index] += r.search(styles.text).group(3)\
                                                            .replace('\r', '').replace('\n', '').replace('\t', '')
                                                    else:
                                                        parts['html_attrs'].extend([
                                                            'style=' + re.sub(
                                                                r'\s{1,}', ' ', r.search(styles.text).group(3).
                                                                replace('\r', '').replace('\n', '').replace('\t', '').strip()
                                                            )
                                                        ])
                                                break
                                if self.tabular:
                                    parent = table_info.parent
                                    parts = table_info.apply_tabular(
                                        parts, parent, self.position)
                                yield Phrase(**parts)
                                self.position += 1
                                self.phrase_num += 1
                            except Exception as e:
                                # This should never happen
                                logger.exception(str(e))

            for child in node:
                if child.tag == 'table':
                    yield from parse_node(
                        child,
                        TableInfo(document=table_info.document),
                        figure_info)
                elif child.tag == 'img':
                    yield from parse_node(
                        child,
                        table_info,
                        FigureInfo(document=figure_info.document))
                else:
                    yield from parse_node(child, table_info, figure_info)

            if self.tabular:
                table_info.exit_tabular(node)

            figure_info.exit_figure(node)

        # Parse document and store text in self.contents, padded with self.delim
        root = fromstring(text)  # lxml.html.fromstring()
        tree = etree.ElementTree(root)
        document.text = text
        yield from parse_node(root, table_info, figure_info)


class TableInfo(object):
    def __init__(self,
                 document,
                 table=None,
                 table_grid=defaultdict(int),
                 cell=None,
                 cell_idx=0,
                 row_idx=0,
                 col_idx=0,
                 parent=None):
        self.document = document
        self.table = table
        self.table_grid = table_grid
        self.cell = cell
        self.cell_idx = cell_idx
        self.row_idx = row_idx
        self.col_idx = col_idx
        self.parent = parent

    def enter_tabular(self, node, table_idx):
        if node.tag == "table":
            table_idx += 1
            self.table_grid.clear()
            self.row_idx = 0
            self.cell_position = 0
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "table", table_idx, table_idx)
            self.table = Table(
                document=self.document,
                stable_id=stable_id,
                position=table_idx)
            self.parent = self.table
        elif node.tag == "tr":
            self.col_idx = 0
        elif node.tag in ["td", "th"]:
            # calculate row_start/col_start
            while self.table_grid[(self.row_idx, self.col_idx)]:
                self.col_idx += 1
            col_start = self.col_idx
            row_start = self.row_idx

            # calculate row_end/col_end
            row_end = row_start
            if "rowspan" in node.attrib:
                row_end += int(node.get("rowspan")) - 1
            col_end = col_start
            if "colspan" in node.attrib:
                col_end += int(node.get("colspan")) - 1

            # update table_grid with occupied cells
            for r, c in itertools.product(
                    list(range(row_start, row_end + 1)),
                    list(range(col_start, col_end + 1))):
                self.table_grid[r, c] = 1

            # construct cell
            parts = defaultdict(list)
            parts["document"] = self.document
            parts["table"] = self.table
            parts["row_start"] = row_start
            parts["row_end"] = row_end
            parts["col_start"] = col_start
            parts["col_end"] = col_end
            parts["position"] = self.cell_position
            parts["stable_id"] = "%s::%s:%s:%s:%s" % (self.document.name,
                                                      "cell",
                                                      self.table.position,
                                                      row_start, col_start)
            self.cell = Cell(**parts)
            self.parent = self.cell
        return table_idx

    def exit_tabular(self, node):
        if node.tag == "table":
            self.table = None
            self.parent = self.document
        elif node.tag == "tr":
            self.row_idx += 1
        elif node.tag in ["td", "th"]:
            self.cell = None
            self.col_idx += 1
            self.cell_idx += 1
            self.cell_position += 1
            self.parent = self.table

    def apply_tabular(self, parts, parent, position):
        parts['position'] = position
        if isinstance(parent, Document):
            pass
        elif isinstance(parent, Table):
            parts['table'] = parent
        elif isinstance(parent, Cell):
            parts['table'] = parent.table
            parts['cell'] = parent
            parts['row_start'] = parent.row_start
            parts['row_end'] = parent.row_end
            parts['col_start'] = parent.col_start
            parts['col_end'] = parent.col_end
        else:
            raise NotImplementedError(
                "Phrase parent must be Document, Table, or Cell")
        return parts


class FigureInfo(object):
    def __init__(self, document, figure=None, parent=None):
        self.document = document
        self.figure = figure
        self.parent = parent

    def enter_figure(self, node, figure_idx):
        if node.tag == "img":
            figure_idx += 1
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "figure", figure_idx, figure_idx)
            self.figure = Figure(
                document=self.document,
                stable_id=stable_id,
                position=figure_idx,
                url=node.get('src'))
            self.parent = self.figure
        return figure_idx

    def exit_figure(self, node):
        if node.tag == "img":
            self.figure = None
            self.parent = self.document
