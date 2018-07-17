import itertools
import logging
import os
import re
from builtins import range
from collections import defaultdict

import lxml

from fonduer.candidates.models import Candidate
from fonduer.parser.models import (
    Cell,
    Context,
    Document,
    Figure,
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


class OmniParser(UDFRunner):
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
            lingual_parser=self.lingual_parser,
        )

    def clear(self, session, **kwargs):
        session.query(Context).delete()

        # We cannot cascade up from child contexts to parent Candidates, so we delete all Candidates too
        session.query(Candidate).delete()


class OmniParserUDF(UDF):
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
        super(OmniParserUDF, self).__init__(**kwargs)

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
            for _ in self.parse_structure(document, text):
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
            yield from self.parse_structure(document, text)

    def _flatten(self, node):
        """Construct a string containing the child's text/tail append to the node.

        If a child of this node is in self.flatten, construct a string
        containing all text/tail results of the tree based on that child and
        append that to the tail of the previous child or head of node.


        :param node: the node to flatten
        """
        num_children = len(node)
        for i, child in enumerate(node[::-1]):
            if child.tag in self.flatten:
                j = num_children - 1 - i  # child index walking backwards
                contents = [""]
                for descendant in child.getiterator():
                    if descendant.text and descendant.text.strip():
                        contents.append(descendant.text)
                    if descendant.tail and descendant.tail.strip():
                        contents.append(descendant.tail)
                if j == 0:
                    if node.text is None:
                        node.text = ""
                    node.text += self.flatten_delim.join(contents)
                else:
                    if node[j - 1].tail is None:
                        node[j - 1].tail = ""
                    node[j - 1].tail += self.flatten_delim.join(contents)
                node.remove(child)

    def _parse_table_node(self, node, state):
        """Parse a table node.

        :param node: The lxml table node to parse
        :param state: The global state necessary to place the node in context
            of the document as a whole.
        """
        if not self.tabular:
            logger.error("Called _parse_table_node without tabular activated.")
            return state

        if node.tag == "table":
            stable_id = "{}::{}:{}".format(
                state["document"].name, "table", state["table"]["idx"]
            )
            # Create the Cell in the DB
            state["context"] = Table(
                document=state["document"],
                stable_id=stable_id,
                position=state["table"]["idx"],
            )

            # Reset variables
            state["table"]["grid"].clear()
            state["table"]["cell"]["row_idx"] = -1
            state["table"]["cell"]["position"] = 0
            state["table"]["idx"] += 1

        elif node.tag == "tr":
            state["table"]["cell"]["col_idx"] = 0
            state["table"]["cell"]["row_idx"] += 1

        elif node.tag in ["td", "th"]:
            # calculate row_start/col_start
            while state["table"]["grid"][
                (state["table"]["cell"]["row_idx"], state["table"]["cell"]["col_idx"])
            ]:
                state["table"]["cell"]["col_idx"] += 1
            col_start = state["table"]["cell"]["col_idx"]
            row_start = state["table"]["cell"]["row_idx"]

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
                state["table"]["grid"][r, c] = 1

            # construct cell
            parts = defaultdict(list)
            parts["document"] = state["document"]
            parts["table"] = state["parent"][node]
            parts["row_start"] = row_start
            parts["row_end"] = row_end
            parts["col_start"] = col_start
            parts["col_end"] = col_end
            parts["position"] = state["table"]["cell"]["position"]
            parts["stable_id"] = "{}::{}:{}:{}:{}".format(
                parts["document"].name,
                "cell",
                parts["table"].position,
                row_start,
                col_start,
            )
            # Create the Cell in the DB
            state["context"] = Cell(**parts)

            # Update position
            state["table"]["cell"]["col_idx"] += 1
            state["table"]["cell"]["position"] += 1

        return state

    def _parse_figure_node(self, node, state):
        """Parse the figure node.

        :param node: The lxml img node to parse
        :param state: The global state necessary to place the node in context
            of the document as a whole.
        """
        if node.tag != "img":
            return state

        # Process the figure
        stable_id = "{}::{}:{}".format(
            state["document"].name, "figure", state["figure"]["idx"]
        )

        # Create the Figure entry in the DB
        # Note that state["context"] is not updated since Figure has no children.
        Figure(
            document=state["document"],
            stable_id=stable_id,
            position=state["figure"]["idx"],
            url=node.get("src"),
        )
        state["figure"]["idx"] += 1

        return state

    def _parse_sentence(self, node, state):
        """Parse the Sentences of the node.

        :param node: The lxml node to parse
        :param state: The global state necessary to place the node in context
            of the document as a whole.
        """
        for field in ["text", "tail"]:
            text = getattr(node, field)
            text = text.strip() if text and self.strip else text

            # Skip if "" or None
            if not text:
                continue

            # Run RegEx replacements
            for (rgx, replace) in self.replacements:
                text = rgx.sub(replace, text)

            # Lingual Parse
            document = state["document"]
            for parts in self.lingual_parse(document, text):
                (_, _, _, char_end) = split_stable_id(parts["stable_id"])
                parts["document"] = document
                parts["sentence_num"] = state["sentence"]["idx"]
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
                    parent = state["parent"][node]
                    if isinstance(parent, Document):
                        pass
                    elif isinstance(parent, Table):
                        parts["table"] = parent
                    elif isinstance(parent, Cell):
                        parts["table"] = parent.table
                        parts["cell"] = parent
                        parts["row_start"] = parent.row_start
                        parts["row_end"] = parent.row_end
                        parts["col_start"] = parent.col_start
                        parts["col_end"] = parent.col_end
                    else:
                        raise NotImplementedError(
                            "Sentence parent must be Document, Table, or Cell"
                        )
                yield Sentence(**parts)
                state["sentence"]["idx"] += 1

    def _parse_node(self, node, state):
        """Entry point for parsing all node types.

        :param node: The lxml HTML node to parse
        :param state: The global state necessary to place the node in context
            of the document as a whole.
        :rtype: a *generator* of Sentences
        """
        # Process node based on its type
        if node.tag is lxml.etree.Comment:
            return
        if self.blacklist and node.tag in self.blacklist:
            return

        # Processing on entry of node
        state = self._parse_figure_node(node, state)

        if self.tabular:
            state = self._parse_table_node(node, state)

        # flattens children of node that are in the 'flatten' list
        if self.flatten:
            self._flatten(node)

        # Now, process the Sentence
        yield from self._parse_sentence(node, state)

    def parse_structure(self, document, text):
        """Depth-first search over the provided tree.

        Implemented as an iterative procedure. The structure of the state
        needed to parse each node is also defined in this function.

        :param document: the Document context
        :param text: the structured text of the document (e.g. HTML)
        :rtype: a *generator* of Sentences.
        """
        stack = []
        visited = set()

        root = lxml.html.fromstring(text)
        document.text = text

        # This dictionary contain the global state necessary to parse a
        # document and each context element. This reflects the relationships
        # defined in the parser/models. This contains the state necessary to
        # create the respective contexts within the document.
        state = {
            "parent": {},  # map of parent[child] = node used to discover child
            "context": None,  # track the most recently created context
            "root": root,
            "document": document,
            "figure": {"idx": 0},
            "table": {
                "grid": defaultdict(int),
                "idx": 0,
                "cell": {"position": 0, "row_idx": -1, "col_idx": 0},
            },
            "sentence": {"idx": 0, "abs_offset": 0},
        }

        # Iterative Depth-First Search
        stack.append(root)
        state["parent"][root] = document
        state["context"] = document
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)  # mark as visited

                # Process
                yield from self._parse_node(node, state)

                # NOTE: This reversed() order is to ensure that the iterative
                # DFS matches the order that would be produced by a recursive
                # DFS implementation.
                for child in reversed(node):
                    stack.append(child)

                    # store the parent of the node
                    state["parent"][child] = state["context"]
