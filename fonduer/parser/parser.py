import itertools
import logging
import os
import re
from builtins import range
from collections import defaultdict

import lxml

from fonduer.candidates.models import Candidate
from fonduer.parser.models import (
    Caption,
    Cell,
    Context,
    Figure,
    Paragraph,
    Section,
    Sentence,
    Table,
    construct_stable_id,
    split_stable_id,
)
from fonduer.parser.simple_tokenizer import SimpleTokenizer
from fonduer.parser.spacy_parser import Spacy
from fonduer.parser.visual_linker import VisualLinker
from fonduer.utils.udf import UDF, UDFRunner

logger = logging.getLogger(__name__)


class Parser(UDFRunner):
    def __init__(
        self,
        structural=True,  # structural information
        blacklist=["style"],  # ignore tag types, default: style
        flatten=["span", "br"],  # flatten tag types, default: span, br
        flatten_delim="",
        lingual=True,  # lingual information
        strip=True,
        replacements=[(u"[\u2010\u2011\u2012\u2013\u2014\u2212\uf02d]", "-")],
        tabular=True,  # tabular information
        visual=False,  # visual information
        pdf_path=None,
    ):
        # Use spaCy as our lingual parser
        self.lingual_parser = Spacy()

        super(Parser, self).__init__(
            ParserUDF,
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
            lingual_parser=self.lingual_parser,
        )

    def clear(self, session, **kwargs):
        session.query(Context).delete()

        # We cannot cascade up from child contexts to parent Candidates, so we delete all Candidates too
        session.query(Candidate).delete()


class ParserUDF(UDF):
    def __init__(
        self,
        structural,
        blacklist,
        flatten,
        flatten_delim,
        lingual,
        strip,
        replacements,
        tabular,
        visual,
        pdf_path,
        lingual_parser,
        **kwargs
    ):
        """
        :param visual: boolean, if True visual features are used in the model
        :param pdf_path: directory where pdf are saved, if a pdf file is not found,
        it will be created from the html document and saved in that directory
        :param replacements: a list of (_pattern_, _replace_) tuples where _pattern_ isinstance
        a regex and _replace_ is a character string. All occurents of _pattern_ in the
        text will be replaced by _replace_.
        """
        super(ParserUDF, self).__init__(**kwargs)

        # structural (html) setup
        self.structural = structural
        self.blacklist = blacklist if isinstance(blacklist, list) else [blacklist]
        self.flatten = flatten if isinstance(flatten, list) else [flatten]
        self.flatten_delim = flatten_delim

        # lingual setup
        self.lingual = lingual
        self.strip = strip
        self.replacements = []
        for (pattern, replace) in replacements:
            self.replacements.append((re.compile(pattern, flags=re.UNICODE), replace))
        if self.lingual:
            self.lingual_parser = lingual_parser
            self.lingual_parse = self.lingual_parser.parse

        else:
            self.lingual_parse = SimpleTokenizer().parse

        # tabular setup
        self.tabular = tabular

        # visual setup
        self.visual = visual
        if self.visual:
            self.pdf_path = pdf_path
            self.vizlink = VisualLinker()

    def apply(self, x, **kwargs):
        # The document is the Document model. Text is string representation.
        document, text = x
        if self.visual:
            if not self.pdf_path:
                logger.error("Visual parsing failed: pdf_path is required")
            for _ in self.parse(document, text):
                pass
            # Add visual attributes
            filename = self.pdf_path + document.name
            missing_pdf = (
                not os.path.isfile(self.pdf_path)
                and not os.path.isfile(filename + ".pdf")
                and not os.path.isfile(filename + ".PDF")
                and not os.path.isfile(filename)
            )
            if missing_pdf:
                logger.error("Visual parsing failed: pdf files are required")
            yield from self.vizlink.parse_visual(
                document.name, document.sentences, self.pdf_path
            )
        else:
            yield from self.parse(document, text)

    def _parse_table(self, node, state):
        """Parse a table node.

        :param node: The lxml table node to parse
        :param state: The global state necessary to place the node in context
            of the document as a whole.
        """
        if not self.tabular:
            logger.error("Called _parse_table without tabular activated.")
            return state

        if node.tag == "table":
            table_idx = state["table"]["idx"]
            stable_id = "{}::{}:{}".format(
                state["document"].name, "table", state["table"]["idx"]
            )
            # Create the Table in the DB
            parts = {}
            parts["document"] = state["document"]
            parts["stable_id"] = stable_id
            parts["position"] = table_idx
            parent = state["parent"][node]
            if isinstance(parent, Cell):
                parts["section"] = parent.table.section
            elif isinstance(parent, Section):
                parts["section"] = parent
            else:
                raise NotImplementedError("Table is not within a Section or Cell")

            state["context"][node] = Table(**parts)

            # Local state for each table. This is required to support nested
            # tables
            state["table"][table_idx] = {
                "grid": defaultdict(int),
                "cell_pos": 0,
                "row_idx": -1,
                "col_idx": 0,
            }

            # Increment table counter
            state["table"]["idx"] += 1

        elif node.tag == "tr":
            if not isinstance(state["parent"][node], Table):
                raise NotImplementedError("Table row parent must be a Table.")

            state["table"][state["parent"][node].position]["col_idx"] = 0
            state["table"][state["parent"][node].position]["row_idx"] += 1

        elif node.tag in ["td", "th"]:
            if not isinstance(state["parent"][node], Table):
                raise NotImplementedError("Cell parent must be a Table.")

            if not state["table"][state["parent"][node].position]["row_idx"] >= 0:
                raise NotImplementedError("Table cell encountered before a table row.")

            # calculate row_start/col_start
            while state["table"][state["parent"][node].position]["grid"][
                (
                    state["table"][state["parent"][node].position]["row_idx"],
                    state["table"][state["parent"][node].position]["col_idx"],
                )
            ]:  # while a cell on the grid is occupied, keep moving
                state["table"][state["parent"][node].position]["col_idx"] += 1
            col_start = state["table"][state["parent"][node].position]["col_idx"]
            row_start = state["table"][state["parent"][node].position]["row_idx"]

            # calculate row_end/col_end
            row_end = row_start
            if "rowspan" in node.attrib:
                row_end += int(node.get("rowspan")) - 1
            col_end = col_start
            if "colspan" in node.attrib:
                col_end += int(node.get("colspan")) - 1

            # update grid with occupied cells
            for r, c in itertools.product(
                list(range(row_start, row_end + 1)), list(range(col_start, col_end + 1))
            ):
                state["table"][state["parent"][node].position]["grid"][(r, c)] = 1

            # construct cell
            parts = defaultdict(list)
            parts["document"] = state["document"]
            parts["table"] = state["parent"][node]
            parts["row_start"] = row_start
            parts["row_end"] = row_end
            parts["col_start"] = col_start
            parts["col_end"] = col_end
            parts["position"] = state["table"][state["parent"][node].position][
                "cell_pos"
            ]
            stable_id = "{}::{}:{}:{}:{}".format(
                parts["document"].name,
                "cell",
                parts["table"].position,
                row_start,
                col_start,
            )
            parts["stable_id"] = stable_id
            # Create the Cell in the DB
            state["context"][node] = Cell(**parts)

            # Update position
            state["table"][state["parent"][node].position]["col_idx"] += 1
            state["table"][state["parent"][node].position]["cell_pos"] += 1

        return state

    def _parse_figure(self, node, state):
        """Parse the figure node.

        :param node: The lxml img node to parse
        :param state: The global state necessary to place the node in context
            of the document as a whole.
        """
        if node.tag not in ["img", "figure"]:
            return state

        # Process the figure
        stable_id = "{}::{}:{}".format(
            state["document"].name, "figure", state["figure"]["idx"]
        )

        # img within a Figure get's processed in the parent Figure
        if node.tag == "img" and isinstance(state["parent"][node], Figure):
            return state

        # NOTE: We currently do NOT support nested figures.
        parts = {}
        parent = state["parent"][node]
        if isinstance(parent, Section):
            parts["section"] = parent
        elif isinstance(parent, Cell):
            parts["section"] = parent.table.section
            parts["cell"] = parent
        else:
            logger.warning("Figure is nested within {}".format(state["parent"][node]))
            return state

        parts["document"] = state["document"]
        parts["stable_id"] = stable_id
        parts["position"] = state["figure"]["idx"]

        # If processing a raw img
        if node.tag == "img":
            # Create the Figure entry in the DB
            parts["url"] = node.get("src")
            state["context"][node] = Figure(**parts)
        elif node.tag == "figure":
            # Pull the image from a child img node, if one exists
            imgs = [child for child in node if child.tag == "img"]

            if len(imgs) > 1:
                logger.warning("Figure contains multiple images.")
                # Right now we don't support multiple URLs in the Figure context
                # As a workaround, just ignore the outer Figure and allow processing
                # of the individual images. We ignore the accompanying figcaption
                # by marking it as visited.
                captions = [child for child in node if child.tag == "figcaption"]
                state["visited"].update(captions)
                return state

            img = imgs[0]
            state["visited"].add(img)

            # Create the Figure entry in the DB
            parts["url"] = img.get("src")
            state["context"][node] = Figure(**parts)

        state["figure"]["idx"] += 1
        return state

    def _parse_sentence(self, paragraph, node, state):
        """Parse the Sentences of the node.

        :param node: The lxml node to parse
        :param state: The global state necessary to place the node in context
            of the document as a whole.
        """
        text = state["paragraph"]["text"]
        field = state["paragraph"]["field"]
        # Lingual Parse
        document = state["document"]
        for parts in self.lingual_parse(document, text):
            (_, _, _, char_end) = split_stable_id(parts["stable_id"])
            parts["document"] = document
            parts["position"] = state["sentence"]["idx"]
            abs_sentence_offset_end = (
                state["sentence"]["abs_offset"]
                + parts["char_offsets"][-1]
                + len(parts["words"][-1])
            )
            parts["stable_id"] = construct_stable_id(
                document,
                "sentence",
                state["sentence"]["abs_offset"],
                abs_sentence_offset_end,
            )
            state["sentence"]["abs_offset"] = abs_sentence_offset_end
            if self.structural:
                context_node = node.getparent() if field == "tail" else node
                tree = lxml.etree.ElementTree(state["root"])
                parts["xpath"] = tree.getpath(context_node)
                parts["html_tag"] = context_node.tag
                parts["html_attrs"] = [
                    "=".join(x) for x in list(context_node.attrib.items())
                ]

                # Extending html style attribute with the styles
                # from inline style class for the element.
                cur_style_index = None
                for index, attr in enumerate(parts["html_attrs"]):
                    if attr.find("style") >= 0:
                        cur_style_index = index
                        break
                styles = state["root"].find("head").find("style")
                if styles is not None:
                    for x in list(context_node.attrib.items()):
                        if x[0] == "class":
                            exp = r"(." + x[1] + ")([\n\s\r]*)\{(.*?)\}"
                            r = re.compile(exp, re.DOTALL)
                            if r.search(styles.text) is not None:
                                if cur_style_index is not None:
                                    parts["html_attrs"][cur_style_index] += (
                                        r.search(styles.text)
                                        .group(3)
                                        .replace("\r", "")
                                        .replace("\n", "")
                                        .replace("\t", "")
                                    )
                                else:
                                    parts["html_attrs"].extend(
                                        [
                                            "style="
                                            + re.sub(
                                                r"\s{1,}",
                                                " ",
                                                r.search(styles.text)
                                                .group(3)
                                                .replace("\r", "")
                                                .replace("\n", "")
                                                .replace("\t", "")
                                                .strip(),
                                            )
                                        ]
                                    )
                            break
            if self.tabular:
                parts["position"] = state["sentence"]["idx"]

                # If tabular, consider own Context first in case a Cell
                # was just created. Otherwise, defer to the parent.
                parent = paragraph
                if isinstance(parent, Paragraph):
                    parts["section"] = parent.section
                    parts["paragraph"] = parent
                    if parent.cell:
                        parts["table"] = parent.cell.table
                        parts["cell"] = parent.cell
                        parts["row_start"] = parent.cell.row_start
                        parts["row_end"] = parent.cell.row_end
                        parts["col_start"] = parent.cell.col_start
                        parts["col_end"] = parent.cell.col_end
                else:
                    raise NotImplementedError("Sentence parent must be Paragraph.")
            yield Sentence(**parts)

            state["sentence"]["idx"] += 1

    def _parse_paragraph(self, node, state):
        """Parse a Paragraph of the node.

        A Paragraph is defined as

        :param node: The lxml node to parse
        :param state: The global state necessary to place the node in context
            of the document as a whole.
        """
        # Both Paragraphs will share the same parent
        parent = (
            state["context"][node]
            if node in state["context"]
            else state["parent"][node]
        )
        for field in ["text", "tail"]:
            text = getattr(node, field)
            text = text.strip() if text and self.strip else text

            # Skip if "" or None
            if not text:
                continue

            # Run RegEx replacements
            for (rgx, replace) in self.replacements:
                text = rgx.sub(replace, text)

            # Process the Paragraph
            stable_id = "{}::{}:{}".format(
                state["document"].name, "paragraph", state["paragraph"]["idx"]
            )
            parts = {}
            parts["stable_id"] = stable_id
            parts["document"] = state["document"]
            parts["position"] = state["paragraph"]["idx"]
            if isinstance(parent, Caption):
                if parent.table:
                    parts["section"] = parent.table.section
                elif parent.figure:
                    parts["section"] = parent.figure.section
                parts["caption"] = parent
            elif isinstance(parent, Cell):
                parts["section"] = parent.table.section
                parts["cell"] = parent
            elif isinstance(parent, Section):
                parts["section"] = parent
            elif isinstance(parent, Figure):  # occurs with text in the tail of an img
                parts["section"] = parent.section
            else:
                raise NotImplementedError(
                    'Paragraph "{}" parent must be Section, Caption, or Cell, not {}'.format(
                        text, parent
                    )
                )

            # Create the Figure entry in the DB
            paragraph = Paragraph(**parts)

            state["paragraph"]["idx"] += 1

            state["paragraph"]["text"] = text
            state["paragraph"]["field"] = field

            # Parse the Sentences in the Paragraph
            yield from self._parse_sentence(paragraph, node, state)

        return state

    def _parse_section(self, node, state):
        """Parse a Section of the node.

        Note that this implementation currently just creates a single Section
        for a document.

        :param node: The lxml node to parse
        :param state: The global state necessary to place the node in context
            of the document as a whole.
        """
        if node.tag != "html":
            return state

        # Add a Section
        stable_id = "{}::{}:{}".format(
            state["document"].name, "section", state["section"]["idx"]
        )
        state["context"][node] = Section(
            document=state["document"],
            stable_id=stable_id,
            position=state["section"]["idx"],
        )
        state["section"]["idx"] += 1

        return state

    def _parse_caption(self, node, state):
        """Parse a Caption of the node.

        :param node: The lxml node to parse
        :param state: The global state necessary to place the node in context
            of the document as a whole.
        """
        if node.tag not in ["caption", "figcaption"]:  # captions used in Tables
            return state

        # Add a Caption
        parent = state["parent"][node]
        stable_id = "{}::{}:{}".format(
            state["document"].name, "caption", state["caption"]["idx"]
        )
        if isinstance(parent, Table):
            state["context"][node] = Caption(
                document=state["document"],
                table=parent,
                figure=None,
                stable_id=stable_id,
                position=state["caption"]["idx"],
            )
        elif isinstance(parent, Figure):
            state["context"][node] = Caption(
                document=state["document"],
                table=None,
                figure=parent,
                stable_id=stable_id,
                position=state["caption"]["idx"],
            )
        else:
            raise NotImplementedError("Caption must be a child of Table or Figure.")
        state["caption"]["idx"] += 1

        return state

    def _parse_node(self, node, state):
        """Entry point for parsing all node types.

        :param node: The lxml HTML node to parse
        :param state: The global state necessary to place the node in context
            of the document as a whole.
        :rtype: a *generator* of Sentences
        """
        # Processing on entry of node
        state = self._parse_section(node, state)

        state = self._parse_figure(node, state)

        if self.tabular:
            state = self._parse_table(node, state)

        state = self._parse_caption(node, state)

        yield from self._parse_paragraph(node, state)

    def parse(self, document, text):
        """Depth-first search over the provided tree.

        Implemented as an iterative procedure. The structure of the state
        needed to parse each node is also defined in this function.

        :param document: the Document context
        :param text: the structured text of the document (e.g. HTML)
        :rtype: a *generator* of Sentences.
        """
        stack = []

        root = lxml.html.fromstring(text)
        document.text = text

        # flattens children of node that are in the 'flatten' list
        if self.flatten:
            lxml.etree.strip_tags(root, self.flatten)

        # This dictionary contain the global state necessary to parse a
        # document and each context element. This reflects the relationships
        # defined in parser/models. This contains the state necessary to create
        # the respective Contexts within the document.
        state = {
            "visited": set(),
            "parent": {},  # map of parent[child] = node used to discover child
            "context": {},  # track the Context created by each node (context['td'] = Cell)
            "root": root,
            "document": document,
            "section": {"idx": 0},
            "paragraph": {"idx": 0},
            "figure": {"idx": 0},
            "caption": {"idx": 0},
            "table": {"idx": 0},
            "sentence": {"idx": 0, "abs_offset": 0},
        }
        # NOTE: Currently the helper functions directly manipulate the state
        # rather than returning a modified copy.

        # Iterative Depth-First Search
        stack.append(root)
        state["parent"][root] = document
        state["context"][root] = document
        while stack:
            node = stack.pop()
            if node not in state["visited"]:
                state["visited"].add(node)  # mark as visited

                # Process
                yield from self._parse_node(node, state)

                # NOTE: This reversed() order is to ensure that the iterative
                # DFS matches the order that would be produced by a recursive
                # DFS implementation.
                for child in reversed(node):
                    # Skip nodes that are comments or blacklisted
                    if child.tag is lxml.etree.Comment or (
                        self.blacklist and child.tag in self.blacklist
                    ):
                        continue

                    stack.append(child)

                    # store the parent of the node, which is either the parent
                    # Context, or if the parent did not create a Context, then
                    # use the node's parent Context.
                    state["parent"][child] = (
                        state["context"][node]
                        if node in state["context"]
                        else state["parent"][node]
                    )
