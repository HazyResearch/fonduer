import codecs
import itertools
import logging
import os
import pdftotree
import re
from builtins import object, range, str
from collections import defaultdict

import numpy as np
from bs4 import BeautifulSoup
from lxml import etree
from lxml.html import fromstring

from fonduer.models import (Table, Cell, Figure, Phrase, Para, Section, Header,
                            FigureCaption, TableCaption, RefList)
from fonduer.snorkel.models import (Candidate, Context, Document,
                                    construct_stable_id, split_stable_id)
from fonduer.snorkel.parser import DocPreprocessor, Spacy
from fonduer.snorkel.udf import UDF, UDFRunner
from fonduer.visual import VisualLinker

logger = logging.getLogger(__name__)


class PDFPreprocessor(DocPreprocessor):
    """Convert all PDF files into HTML using pdftotree.

    Then, yield a Document for each resulting HTML.
    """

    def parse_file(self, fp, file_name):
        # Use pdftotree to convert the file to HTML
        logger.debug("Converting {} to HTML using pdftotree...".format(fp))
        try:
            html = pdftotree.parse(fp)
            name = os.path.basename(fp)[:os.path.basename(fp).rfind('.')]
            stable_id = self.get_stable_id(name)
            logger.debug("Yielding {}.".format(stable_id))
            yield Document(
                name=name,
                stable_id=stable_id,
                text=str(html),
                meta={
                    'file_name': file_name
                }), str(html)
        except Exception as e:
            print("{}".format(fp))

    def _can_read(self, fpath):
        return fpath.upper().endswith('.PDF')


class HTMLPreprocessor(DocPreprocessor):
    """Simple parsing of files into html documents"""

    def parse_file(self, fp, file_name):
        with codecs.open(fp, encoding=self.encoding) as f:
            soup = BeautifulSoup(f, 'lxml')
            for text in soup.find_all('html'):
                name = os.path.basename(fp)[:os.path.basename(fp).rfind('.')]
                stable_id = self.get_stable_id(name)
                yield Document(
                    name=name,
                    stable_id=stable_id,
                    text=str(text),
                    meta={
                        'file_name': file_name
                    }), str(text)

    def _can_read(self, fpath):
        return fpath.endswith('html')  # includes both .html and .xhtml


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
            char_offsets = [0] + list(np.cumsum([len(x) + 1
                                                 for x in words]))[:-1]
            text = ' '.join(words)
            stable_id = construct_stable_id(document, 'phrase', i, i)
            yield {
                'text': text,
                'words': words,
                'char_offsets': char_offsets,
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
            replacements=[('\(cid:\d+\)', '$')],  #[(u'[\u2010\u2011\u2012\u2013\u2014\u2212\uf02d]', '-')],
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
            self.req_handler = lingual_parser.connect()
            self.lingual_parse = self.req_handler.parse

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

    def _add_word_coordinates(self, parts):
        """Get word-level coordinates for each word.

        pdftotree outputs character level coordinates. This function updates
        adds word-level coordinates to match the words output by spaCy.
        """
        # Turn the html_attributes into a more usable dictionary of lists. We
        # only split on the first leftmost equal sign to allow '=' to appear as
        # a char itself.
        coords = dict()
        for html_attr in [_.split('=', 1) for _ in parts['html_attrs']]:
            # NOTE: The replacing of double spaces is used when a ' ' is a
            # character itself. Otherwise '  '.split() = ['', '']
            temp = html_attr[1].strip().replace('  ', ' ').split(' ')
            # Next, we need to replace the '' for spaces with a space
            coords[html_attr[0]] = [' ' if _ == '' else _ for _ in temp]

        # Simple approach to matching up words with their characters. We
        # advance this idx as we parse through the string. For example,
        # parts['words'] = ['Types', 'of', 'viruses', ',', 'coughs', ',', 'and', 'colds']
        # char=T y p e s \xa0 o f \xa0 v i r u s e s , \xa0 c o u g h s , \xa0 a n d \xa0 c o l d s '
        char_idx = 0
        for word in parts['words']:
            curr_word = [
                word,
                float('Inf'),
                float('Inf'),
                float('-Inf'),
                float('-Inf')
            ]
            # We don't use parts['text'] here because it is not exactly mapped
            # to the coordinates provided by pdftotree. For example, '\xa0' is
            # given a coordinate, while a normal space is not.
            try:
                char_str = ''.join(coords['char'])
                for (rgx, replace) in self.replacements:
                    char_str = rgx.sub(replace, char_str)
                start = char_str.find(curr_word[0], char_idx)
                end = start + len(curr_word[0])

                if start == -1:
                    msg = "\"{}\" was not found in \"{}\"".format(
                        curr_word[0], char_str[char_idx:])
                    raise ValueError(msg)

                for char_iter in range(start, end):
                    curr_word[1] = int(
                        min(curr_word[1], float(coords['top'][char_iter])))
                    curr_word[2] = int(
                        min(curr_word[2], float(coords['left'][char_iter])))
                    curr_word[3] = int(
                        max(curr_word[3], float(coords['bottom'][char_iter])))
                    curr_word[4] = int(
                        max(curr_word[4], float(coords['right'][char_iter])))
                char_idx = end

                parts['top'].append(curr_word[1])
                parts['left'].append(curr_word[2])
                parts['bottom'].append(curr_word[3])
                parts['right'].append(curr_word[4])
            except Exception as e:
                logger.exception(e)


        # TODO(lwhsiao): I think this actually is uncessesary, as the parts
        # object gets modified by this function directly. But, it might be nice
        # in that it's a bit more clear this way.
        return parts

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

        para_info = ParaInfo(document, parent=document)
        self.para_idx = -1
        parents_para = []

        section_info = SectionInfo(document, parent=document)
        self.section_idx = -1
        parents_section = []

        header_info = HeaderInfo(document, parent=document)
        self.header_idx = -1
        parents_header = []

        figCaption_info = FigureCaptionInfo(document, parent=document)
        self.figCaption_idx = -1
        parents_figCaption = []

        tabCaption_info = TableCaptionInfo(document, parent=document)
        self.tabCaption_idx = -1
        parents_tabCaption = []

        refList_info = RefListInfo(document, parent=document)
        self.refList_idx = -1
        parents_refList = []

        self.coordinates = {}
        self.char_idx = {}

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

        def parse_node(node,
                       table_info=None,
                       figure_info=None,
                       para_info=None,
                       section_info=None,
                       header_info=None,
                       figCaption_info=None,
                       tabCaption_info=None,
                       refList_info=None):
            if node.tag is etree.Comment:
                return
            if self.blacklist and node.tag in self.blacklist:
                return

            self.para_idx, coordinates = para_info.enter_para(
                node, self.para_idx, {})
            if len(coordinates) > 0:
                self.coordinates[self.para_idx] = coordinates
                self.char_idx[self.para_idx] = 0

            self.figure_idx = figure_info.enter_figure(node, self.figure_idx)

            self.section_idx, self.para_idx, coordinates = section_info.enter_section(
                node, self.section_idx, self.para_idx, {})
            if len(coordinates) > 0:
                self.coordinates[self.para_idx] = coordinates
                self.char_idx[self.para_idx] = 0
            self.header_idx, self.para_idx, coordinates = header_info.enter_header(
                node, self.header_idx, self.para_idx, {})
            if len(coordinates) > 0:
                self.coordinates[self.para_idx] = coordinates
                self.char_idx[self.para_idx] = 0
            self.figCaption_idx, self.para_idx, coordinates = figCaption_info.enter_figCaption(
                node, self.figCaption_idx, self.para_idx, {})
            if len(coordinates) > 0:
                self.coordinates[self.para_idx] = coordinates
                self.char_idx[self.para_idx] = 0
            self.tabCaption_idx, self.para_idx, coordinates = tabCaption_info.enter_tabCaption(
                node, self.tabCaption_idx, self.para_idx, {})
            if len(coordinates) > 0:
                self.coordinates[self.para_idx] = coordinates
                self.char_idx[self.para_idx] = 0
            self.refList_idx, self.para_idx, coordinates = refList_info.enter_refList(
                node, self.refList_idx, self.para_idx, {})
            if len(coordinates) > 0:
                self.coordinates[self.para_idx] = coordinates
                self.char_idx[self.para_idx] = 0

            if self.tabular:
                self.table_idx, self.para_idx, coordinates = table_info.enter_tabular(
                    node, self.table_idx, self.para_idx, {})
                if len(coordinates) > 0:
                    self.coordinates[self.para_idx] = coordinates
                    self.char_idx[self.para_idx] = 0

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

                        parents_para.append(para_info.parent)
                        parents_section.append(section_info.parent)
                        parents_header.append(header_info.parent)
                        parents_figCaption.append(figCaption_info.parent)
                        parents_tabCaption.append(tabCaption_info.parent)
                        parents_refList.append(refList_info.parent)

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

                                # Update the coordinates using information from
                                # pdftotree directly when parsing the text.
                                parts = self._add_word_coordinates(parts)

                                parent = parents_para[self.parent_idx]
                                parts, self.char_idx = para_info.apply_para(
                                    parts, parent, self.position,
                                    self.coordinates, self.char_idx)

                                parent = parents_section[self.parent_idx]
                                parts, self.char_idx = section_info.apply_section(
                                    parts, parent, self.position,
                                    self.coordinates, self.char_idx)

                                parent = parents_header[self.parent_idx]
                                parts, self.char_idx = header_info.apply_header(
                                    parts, parent, self.position,
                                    self.coordinates, self.char_idx)

                                parent = parents_figCaption[self.parent_idx]
                                parts, self.char_idx = figCaption_info.apply_figCaption(
                                    parts, parent, self.position,
                                    self.coordinates, self.char_idx)

                                parent = parents_tabCaption[self.parent_idx]
                                parts, self.char_idx = tabCaption_info.apply_tabCaption(
                                    parts, parent, self.position,
                                    self.coordinates, self.char_idx)

                                parent = parents_refList[self.parent_idx]
                                parts, self.char_idx = refList_info.apply_refList(
                                    parts, parent, self.position,
                                    self.coordinates, self.char_idx)

                                if self.tabular:
                                    parent = table_info.parent
                                    parts = table_info.apply_tabular(
                                        parts, parent, self.position,
                                        self.coordinates)
                                yield Phrase(**parts)
                                self.position += 1
                                self.phrase_num += 1
                            except Exception as e:
                                # This should never happen
                                logger.warning("{}".format(document))
                                logger.exception(e)

            for child in node:
                if child.tag == 'table':
                    yield from parse_node(
                        child,
                        TableInfo(document=table_info.document),
                        figure_info,
                        para_info,
                        section_info,
                        header_info,
                        figCaption_info,
                        tabCaption_info,
                        refList_info)
                elif child.tag == 'img' or child.tag == 'figure':
                    yield from parse_node(
                        child,
                        table_info,
                        FigureInfo(document=figure_info.document),
                        para_info,
                        section_info,
                        header_info,
                        figCaption_info,
                        tabCaption_info,
                        refList_info)
                elif child.tag == 'paragraph':
                    yield from parse_node(
                        child,
                        table_info,
                        figure_info,
                        ParaInfo(document=para_info.document),
                        section_info,
                        header_info,
                        figCaption_info,
                        tabCaption_info,
                        refList_info)
                elif child.tag == 'section_header':
                    yield from parse_node(
                        child,
                        table_info,
                        figure_info,
                        para_info,
                        SectionInfo(document=section_info.document),
                        header_info,
                        figCaption_info,
                        tabCaption_info,
                        refList_info)
                elif child.tag == 'header':
                    yield from parse_node(
                        child,
                        table_info,
                        figure_info,
                        para_info,
                        section_info,
                        HeaderInfo(document=header_info.document),
                        figCaption_info,
                        tabCaption_info,
                        refList_info)
                elif child.tag == 'figure_caption':
                    yield from parse_node(
                        child,
                        table_info,
                        figure_info,
                        para_info,
                        section_info,
                        header_info,
                        FigureCaptionInfo(document=figCaption_info.document),
                        tabCaption_info,
                        refList_info)
                elif child.tag == 'table_caption':
                    yield from parse_node(
                        child,
                        table_info,
                        figure_info,
                        para_info,
                        section_info,
                        header_info,
                        figCaption_info,
                        TableCaptionInfo(document=tabCaption_info.document),
                        refList_info)
                elif child.tag == 'list':
                    yield from parse_node(
                        child,
                        table_info,
                        figure_info,
                        para_info,
                        section_info,
                        header_info,
                        figCaption_info,
                        tabCaption_info,
                        RefListInfo(document=refList_info.document))
                else:
                    yield from parse_node(child, table_info, figure_info,
                                          para_info, section_info, header_info,
                                          figCaption_info, tabCaption_info,
                                          refList_info)

            if self.tabular:
                table_info.exit_tabular(node)

            refList_info.exit_refList(node)
            tabCaption_info.exit_tabCaption(node)
            figCaption_info.exit_figCaption(node)
            header_info.exit_header(node)
            section_info.exit_section(node)
            para_info.exit_para(node)
            figure_info.exit_figure(node)

        # Parse document and store text in self.contents, padded with self.delim
        root = fromstring(text)  # lxml.html.fromstring()
        tree = etree.ElementTree(root)
        document.text = text
        yield from parse_node(root, table_info, figure_info, para_info,
                              section_info, header_info, figCaption_info,
                              tabCaption_info, refList_info)


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

    def enter_tabular(self, node, table_idx, para_idx, coordinates):
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

            # construct para
            para_idx += 1
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "para_cell", para_idx, para_idx)
            self.para = Para(
                document=self.document, stable_id=stable_id, position=para_idx)

            # construct cell
            parts = defaultdict(list)
            parts["document"] = self.document
            parts["table"] = self.table
            parts["para"] = self.para
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

            # coordinates
            coordinates = {}
            coordinates["word"] = node.get('word')
            coordinates["top"] = node.get('top')
            coordinates["left"] = node.get('left')
            coordinates["bottom"] = node.get('bottom')
            coordinates["right"] = node.get('right')

        return table_idx, para_idx, coordinates

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

    def apply_tabular(self, parts, parent, position, coordinates):
        parts['position'] = position
        if isinstance(parent, Document):
            pass
        elif isinstance(parent, Table):
            parts['table'] = parent
        elif isinstance(parent, Cell):
            parts['table'] = parent.table
            parts['cell'] = parent
            parts['para'] = parent.para
            parts['row_start'] = parent.row_start
            parts['row_end'] = parent.row_end
            parts['col_start'] = parent.col_start
            parts['col_end'] = parent.col_end
            #  parts = update_coordinates_table(parts,
            #                                   coordinates[parent.para.position])
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
        if node.tag == "img" or node.tag == "figure":
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
        if node.tag == "img" or node.tag == "figure":
            self.figure = None
            self.parent = self.document


class ParaInfo(object):
    def __init__(self, document, para=None, parent=None):
        self.document = document
        self.para = para
        self.parent = parent

    def enter_para(self, node, para_idx, coordinates):
        if node.tag == "paragraph":
            para_idx += 1
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "para", para_idx, para_idx)
            self.para = Para(
                document=self.document, stable_id=stable_id, position=para_idx)
            self.parent = self.para
            #coordinates
            coordinates = {}
            coordinates["char"] = node.get('char')
            coordinates["top"] = node.get('top')
            coordinates["left"] = node.get('left')
            coordinates["bottom"] = node.get('bottom')
            coordinates["right"] = node.get('right')
        return para_idx, coordinates

    def exit_para(self, node):
        if node.tag == "paragraph":
            self.para = None
            self.parent = self.document

    def apply_para(self, parts, parent, position, coordinates, char_idx):
        parts['position'] = position
        if isinstance(parent, Document):
            pass
        elif isinstance(parent, Para):
            parts['para'] = parent
            #  parts, char_idx[parent.position] = update_coordinates(
            #      parts, coordinates[parent.position], char_idx[parent.position])
        else:
            raise NotImplementedError("Phrase parent must be Document or Para")
        return parts, char_idx


def update_coordinates_table(parts, coordinates):
    sep = " "
    words = coordinates["word"][:-1].split(sep)
    top = [float(_) for _ in coordinates["top"][:-1].split(sep)]
    left = [float(_) for _ in coordinates["left"][:-1].split(sep)]
    bottom = [float(_) for _ in coordinates["bottom"][:-1].split(sep)]
    right = [float(_) for _ in coordinates["right"][:-1].split(sep)]
    max_len = len(words)
    i = 0
    for word in parts["words"]:
        parts['top'].append(top)
        parts['left'].append(left)
        parts['bottom'].append(bottom)
        parts['right'].append(right)
        i += 1
        if i == max_len:
            break
    return parts


def lcs(X, Y):
    m = len(X)
    n = len(Y)

    L = [[None] * (n + 1) for i in range(m + 1)]
    d = [[None] * (n + 1) for i in range(m + 1)]
    """Following steps build L[m+1][n+1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    matches = []
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1] and (L[i - 1][j - 1] + 1) > max(
                    L[i - 1][j], L[i][j - 1]):
                L[i][j] = L[i - 1][j - 1] + 1
                d[i][j] = 'd'
            else:
                if L[i][j - 1] > L[i - 1][j]:
                    d[i][j] = 'u'
                    L[i][j] = L[i][j - 1]
                else:
                    d[i][j] = 'l'
                    L[i][j] = L[i - 1][j]
    i = m
    j = n
    while i >= 0 and j >= 0:
        if d[i][j] == 'u':
            j -= 1
        elif d[i][j] == 'l':
            i -= 1
        else:
            matches.append((i, j))
            i -= 1
            j -= 1
    return matches


def update_coordinates(parts, coordinates, char_idx):
    sep = " "
    chars = coordinates['char'][:-1].split(sep)
    top = [float(_) for _ in coordinates['top'][:-1].split(sep)]
    left = [float(_) for _ in coordinates['left'][:-1].split(sep)]
    bottom = [float(_) for _ in coordinates['bottom'][:-1].split(sep)]
    right = [float(_) for _ in coordinates['right'][:-1].split(sep)]
    words = []
    new_chars = []
    new_top = []
    new_left = []
    new_bottom = []
    new_right = []
    for i, char in enumerate(chars):
        if len(char) > 0:
            new_chars.append(char)
            new_top.append(top[i])
            new_left.append(left[i])
            new_bottom.append(bottom[i])
            new_right.append(right[i])
    chars = new_chars
    top = new_top
    left = new_left
    right = new_right
    bottom = new_bottom
    words = []
    matches = lcs("".join(chars[char_idx:]), "".join(parts['words']))
    word_lens = [len(words) for words in parts['words']]
    logger.debug("parts['words'] = {}".format(parts['words']))
    for i, word in enumerate(parts['words']):
        curr_word = [
            word,
            float('Inf'),
            float('Inf'),
            float('-Inf'),
            float('-Inf')
        ]
        word_len = 0
        word_len += sum(word_lens[:i])
        word_begin = -1
        word_end = -1
        for match in matches:
            if match[1] == word_len:
                word_begin = match[0]
            if match[1] == word_len + word_lens[i]:
                word_end = match[0]
        if word_begin == -1 or word_end == -1:
            logger.warning("no match found")
        else:
            for char_iter in range(word_begin, word_end):
                curr_word[1] = int(
                    min(curr_word[1], top[char_idx + char_iter]))
                curr_word[2] = int(
                    min(curr_word[2], left[char_idx + char_iter]))
                curr_word[3] = int(
                    max(curr_word[3], bottom[char_idx + char_iter]))
                curr_word[4] = int(
                    max(curr_word[4], right[char_idx + char_iter]))
        parts['top'].append(curr_word[1])
        parts['left'].append(curr_word[2])
        parts['bottom'].append(curr_word[3])
        parts['right'].append(curr_word[4])
    char_idx += max([x[0] for x in matches])
    return parts, char_idx


class SectionInfo(object):
    def __init__(self, document, section=None, parent=None):
        self.document = document
        self.section = section
        self.parent = parent

    def enter_section(self, node, section_idx, para_idx, coordinates):
        if node.tag == "section_header":
            para_idx += 1
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "para_section", para_idx, para_idx)
            para = Para(
                document=self.document, stable_id=stable_id, position=para_idx)

            section_idx += 1
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "section", section_idx, section_idx)
            self.section = Section(
                document=self.document,
                stable_id=stable_id,
                position=section_idx,
                para=para,
                para_id=para_idx)
            self.parent = self.section
            #coordinates
            coordinates = {}
            coordinates["char"] = node.get('char')
            coordinates["top"] = node.get('top')
            coordinates["left"] = node.get('left')
            coordinates["bottom"] = node.get('bottom')
            coordinates["right"] = node.get('right')
        return section_idx, para_idx, coordinates

    def exit_section(self, node):
        if node.tag == "section_header":
            self.section = None
            self.parent = self.document

    def apply_section(self, parts, parent, position, coordinates, char_idx):
        parts['position'] = position
        if isinstance(parent, Document):
            pass
        elif isinstance(parent, Section):
            parts['para'] = parent.para
            #  parts, char_idx[parent.para.position] = update_coordinates(
            #      parts, coordinates[parent.para.position],
            #      char_idx[parent.para.position])
        else:
            raise NotImplementedError(
                "Phrase parent must be Document or Section")
        return parts, char_idx


class HeaderInfo(object):
    def __init__(self, document, header=None, parent=None):
        self.document = document
        self.header = header
        self.parent = parent

    def enter_header(self, node, header_idx, para_idx, coordinates):
        if node.tag == "header":
            para_idx += 1
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "para_header", para_idx, para_idx)
            para = Para(
                document=self.document, stable_id=stable_id, position=para_idx)

            header_idx += 1
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "header", header_idx, header_idx)
            self.header = Header(
                document=self.document,
                stable_id=stable_id,
                position=header_idx,
                para=para)
            self.parent = self.header
            #coordinates
            coordinates = {}
            coordinates["char"] = node.get('char')
            coordinates["top"] = node.get('top')
            coordinates["left"] = node.get('left')
            coordinates["bottom"] = node.get('bottom')
            coordinates["right"] = node.get('right')
        return header_idx, para_idx, coordinates

    def exit_header(self, node):
        if node.tag == "header":
            self.header = None
            self.parent = self.document

    def apply_header(self, parts, parent, position, coordinates, char_idx):
        parts['position'] = position
        if isinstance(parent, Document):
            pass
        elif isinstance(parent, Header):
            parts['para'] = parent.para
            #  parts, char_idx[parent.para.position] = update_coordinates(
            #      parts, coordinates[parent.para.position],
            #      char_idx[parent.para.position])
        else:
            raise NotImplementedError(
                "Phrase parent must be Document or Header")
        return parts, char_idx


class FigureCaptionInfo(object):
    def __init__(self, document, figCaption=None, parent=None):
        self.document = document
        self.figCaption = figCaption
        self.parent = parent

    def enter_figCaption(self, node, figCaption_idx, para_idx, coordinates):
        if node.tag == "figure_caption":
            para_idx += 1
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "para_figCaption", para_idx, para_idx)
            para = Para(
                document=self.document, stable_id=stable_id, position=para_idx)

            figCaption_idx += 1
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "figCaption", figCaption_idx, figCaption_idx)
            self.figCaption = FigureCaption(
                document=self.document,
                stable_id=stable_id,
                position=figCaption_idx,
                para=para)
            self.parent = self.figCaption
            #coordinates
            coordinates = {}
            coordinates["char"] = node.get('char')
            coordinates["top"] = node.get('top')
            coordinates["left"] = node.get('left')
            coordinates["bottom"] = node.get('bottom')
            coordinates["right"] = node.get('right')
        return figCaption_idx, para_idx, coordinates

    def exit_figCaption(self, node):
        if node.tag == "figure_caption":
            self.figCaption = None
            self.parent = self.document

    def apply_figCaption(self, parts, parent, position, coordinates, char_idx):
        parts['position'] = position
        if isinstance(parent, Document):
            pass
        elif isinstance(parent, FigureCaption):
            parts['para'] = parent.para
            #  parts, char_idx[parent.para.position] = update_coordinates(
            #      parts, coordinates[parent.para.position],
            #      char_idx[parent.para.position])
        else:
            raise NotImplementedError(
                "Phrase parent must be Document or FigureCaption")
        return parts, char_idx


class TableCaptionInfo(object):
    def __init__(self, document, tabCaption=None, parent=None):
        self.document = document
        self.tabCaption = tabCaption
        self.parent = parent

    def enter_tabCaption(self, node, tabCaption_idx, para_idx, coordinates):
        if node.tag == "table_caption":
            para_idx += 1
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "para_tabCaption", para_idx, para_idx)
            para = Para(
                document=self.document, stable_id=stable_id, position=para_idx)

            tabCaption_idx += 1
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "tabCaption", tabCaption_idx, tabCaption_idx)
            self.tabCaption = TableCaption(
                document=self.document,
                stable_id=stable_id,
                position=tabCaption_idx,
                para=para)
            self.parent = self.tabCaption
            #coordinates
            coordinates = {}
            coordinates["char"] = node.get('char')
            coordinates["top"] = node.get('top')
            coordinates["left"] = node.get('left')
            coordinates["bottom"] = node.get('bottom')
            coordinates["right"] = node.get('right')
        return tabCaption_idx, para_idx, coordinates

    def exit_tabCaption(self, node):
        if node.tag == "table_caption":
            self.tabCaption = None
            self.parent = self.document

    def apply_tabCaption(self, parts, parent, position, coordinates, char_idx):
        parts['position'] = position
        if isinstance(parent, Document):
            pass
        elif isinstance(parent, TableCaption):
            parts['para'] = parent.para
            #  parts, char_idx[parent.para.position] = update_coordinates(
            #      parts, coordinates[parent.para.position],
            #      char_idx[parent.para.position])
        else:
            raise NotImplementedError(
                "Phrase parent must be Document or TableCaption")
        return parts, char_idx


class RefListInfo(object):
    def __init__(self, document, refList=None, parent=None):
        self.document = document
        self.refList = refList
        self.parent = parent

    def enter_refList(self, node, refList_idx, para_idx, coordinates):
        if node.tag == "list":
            para_idx += 1
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "para_refList", para_idx, para_idx)
            para = Para(
                document=self.document, stable_id=stable_id, position=para_idx)

            refList_idx += 1
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "refList", refList_idx, refList_idx)
            self.refList = RefList(
                document=self.document,
                stable_id=stable_id,
                position=refList_idx,
                para=para)
            self.parent = self.refList
            #coordinates
            coordinates = {}
            coordinates["char"] = node.get('char')
            coordinates["top"] = node.get('top')
            coordinates["left"] = node.get('left')
            coordinates["bottom"] = node.get('bottom')
            coordinates["right"] = node.get('right')
        return refList_idx, para_idx, coordinates

    def exit_refList(self, node):
        if node.tag == "list":
            self.refList = None
            self.parent = self.document

    def apply_refList(self, parts, parent, position, coordinates, char_idx):
        parts['position'] = position
        if isinstance(parent, Document):
            pass
        elif isinstance(parent, RefList):
            parts['para'] = parent.para
            parts, char_idx[parent.para.position] = update_coordinates(
                parts, coordinates[parent.para.position],
                char_idx[parent.para.position])
        else:
            raise NotImplementedError(
                "Phrase parent must be Document or RefList")
        return parts, char_idx
