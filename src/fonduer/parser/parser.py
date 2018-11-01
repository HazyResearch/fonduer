import itertools
import logging
import os
import re
import warnings
from builtins import range
from collections import defaultdict

import lxml.etree
import lxml.html

from fonduer.parser.models import (
    Caption,
    Cell,
    Context,
    Document,
    Figure,
    Paragraph,
    Section,
    Sentence,
    Table,
)
from fonduer.parser.models.utils import construct_stable_id
from fonduer.parser.simple_tokenizer import SimpleTokenizer
from fonduer.parser.spacy_parser import Spacy
from fonduer.parser.visual_linker import VisualLinker
from fonduer.utils.udf import UDF, UDFRunner

logger = logging.getLogger(__name__)


class Parser(UDFRunner):
    """Parses into documents into Fonduer's Data Model.

    :param session: The database session to use.
    :param parallelism: The number of processes to use in parallel. Default 1.
    :param structural: Whether to parse structural information from a DOM.
    :param blacklist: A list of tag types to ignore. Default ["style", "script"].
    :param flatten: A list of tag types to flatten. Default ["span", "br"]
    :param language: Which spaCy NLP language package. Default "en".
    :param lingual: Whether or not to include NLP information. Default True.
    :param strip: Whether or not to strip whitespace during parsing. Default True.
    :param replacements: A list of tuples where the regex string in the
        first position is replaced by the character in the second position.
        Default [(u"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-")], which
        replaces various unicode variants of a hyphen (e.g. emdash, endash,
        minus, etc.) with a standard ASCII hyphen.
    :param tabular: Whether to include tabular information in the parse.
    :param visual: Whether to include visual information in the parse.
        Requires PDFs for each input document.
    :param pdf_path: The path to the corresponding PDFs use for visual info.
    """

    def __init__(
        self,
        session,
        parallelism=1,
        structural=True,  # structural information
        blacklist=["style", "script"],  # ignore tag types, default: style, script
        flatten=["span", "br"],  # flatten tag types, default: span, br
        language="en",
        lingual=True,  # lingual information
        strip=True,
        replacements=[(u"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-")],
        tabular=True,  # tabular information
        visual=False,  # visual information
        pdf_path=None,
    ):
        super(Parser, self).__init__(
            session,
            ParserUDF,
            parallelism=parallelism,
            structural=structural,
            blacklist=blacklist,
            flatten=flatten,
            lingual=lingual,
            strip=strip,
            replacements=replacements,
            tabular=tabular,
            visual=visual,
            pdf_path=pdf_path,
            language=language,
        )

    def apply(
        self, doc_loader, pdf_path=None, clear=True, parallelism=None, progress_bar=True
    ):
        """Run the Parser.

        :param doc_loader: An iteratable of ``Documents`` to parse. Typically,
            one of Fonduer's document preprocessors.
        :param pdf_path: The path to the PDF documents, if any. This path will
            override the one used in initialization, if provided.
        :param clear: Whether or not to clear the labels table before applying
            these LFs.
        :type clear: bool
        :param parallelism: How many threads to use for extraction. This will
            override the parallelism value used to initialize the Labeler if
            it is provided.
        :type parallelism: int
        :param progress_bar: Whether or not to display a progress bar. The
            progress bar is measured per document.
        :type progress_bar: bool
        """
        super(Parser, self).apply(
            doc_loader,
            pdf_path=pdf_path,
            clear=clear,
            parallelism=parallelism,
            progress_bar=progress_bar,
        )

    def clear(self, pdf_path=None):
        """Clear all of the ``Context`` objects in the database.

        :param pdf_path: This parameter is ignored.
        """
        self.session.query(Context).delete()

    def get_documents(self):
        """Return all the parsed ``Documents`` in the database.

        :rtype: A list of all ``Documents`` in the database ordered by name.
        """
        return self.session.query(Document).order_by(Document.name).all()


class ParserUDF(UDF):
    def __init__(
        self,
        structural,
        blacklist,
        flatten,
        lingual,
        strip,
        replacements,
        tabular,
        visual,
        pdf_path,
        language,
        **kwargs
    ):
        """
        :param visual: boolean, if True visual features are used in the model
        :param pdf_path: directory where pdf are saved, if a pdf file is not
            found, it will be created from the html document and saved in that
            directory
        :param replacements: a list of (_pattern_, _replace_) tuples where
            _pattern_ isinstance a regex and _replace_ is a character string.
            All occurents of _pattern_ in the text will be replaced by
            _replace_.
        """
        super(ParserUDF, self).__init__(**kwargs)

        # structural (html) setup
        self.structural = structural
        self.blacklist = blacklist if isinstance(blacklist, list) else [blacklist]
        self.flatten = flatten if isinstance(flatten, list) else [flatten]

        # lingual setup
        self.language = language
        self.strip = strip
        self.replacements = []
        for (pattern, replace) in replacements:
            self.replacements.append((re.compile(pattern, flags=re.UNICODE), replace))

        self.lingual = lingual
        self.lingual_parser = Spacy(self.language)
        if self.lingual_parser.has_tokenizer_support():
            self.tokenize_and_split_sentences = self.lingual_parser.split_sentences
            self.lingual_parser.load_lang_model()
        else:
            self.tokenize_and_split_sentences = SimpleTokenizer().parse

        if self.lingual:
            if self.lingual_parser.has_NLP_support():
                self.enrich_tokenized_sentences_with_nlp = (
                    self.lingual_parser.enrich_sentences_with_NLP
                )
            else:
                logger.warning(
                    "Lingual mode will be turned off, "
                    "as spacy doesn't provide support for this "
                    "language ({})".format(self.language)
                )
                self.lingual = False

        # tabular setup
        self.tabular = tabular

        # visual setup
        self.visual = visual
        if self.visual:
            self.pdf_path = pdf_path
            self.vizlink = VisualLinker()

    def apply(self, document, pdf_path=None, **kwargs):
        # The document is the Document model
        text = document.text

        # Only return sentences, if no exceptions occur during parsing
        try:
            return_sentences = []
            if self.visual:
                # Use the provided pdf_path if present
                self.pdf_path = pdf_path if pdf_path else self.pdf_path
                if not self.pdf_path:
                    warnings.warn(
                        "Visual parsing failed: pdf_path is required. "
                        + "Proceeding without visual parsing.",
                        RuntimeWarning,
                    )
                    self.visual = False
                    return_sentences += [y for y in self.parse(document, text)]

                elif not self._valid_pdf(self.pdf_path, document.name):
                    warnings.warn(
                        "Visual parse failed. {} not a PDF. {}".format(
                            self.pdf_path + document.name,
                            "Proceeding without visual parsing.",
                        ),
                        RuntimeWarning,
                    )
                    self.visual = False
                    return_sentences += [y for y in self.parse(document, text)]
                else:
                    # Populate document.sentences
                    for _ in self.parse(document, text):
                        pass
                    # Add visual attributes
                    return_sentences += [
                        y
                        for y in self.vizlink.parse_visual(
                            document.name, document.sentences, self.pdf_path
                        )
                    ]
            else:
                return_sentences += [y for y in self.parse(document, text)]

            yield from return_sentences
        except NotImplementedError as e:
            warnings.warn(
                "Document {} not added to database, "
                "because of parse error: \n{}".format(document.name, e)
            )

    def _valid_pdf(self, path, filename):
        """Verify that the file exists and has a PDF extension."""
        # If path is file, but not PDF.
        if os.path.isfile(path) and path.lower().endswith(".pdf"):
            return True
        else:
            full_path = os.path.join(path, filename)
            if os.path.isfile(full_path) and full_path.lower().endswith(".pdf"):
                return True
            elif os.path.isfile(os.path.join(path, filename + ".pdf")):
                return True
            elif os.path.isfile(os.path.join(path, filename + ".PDF")):
                return True

        return False

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
        for parts in self.tokenize_and_split_sentences(document, text):
            parts["document"] = document
            # NOTE: Why do we overwrite this from the spacy parse?
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
                head = state["root"].find("head")
                styles = None
                if head is not None:
                    styles = head.find("style")
                if styles is not None:
                    for x in list(context_node.attrib.items()):
                        if x[0] == "class":
                            exp = r"(." + x[1] + r")([\n\s\r]*)\{(.*?)\}"
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
                    (
                        'Para "{}" parent must be Section, Caption, or Cell, not {}'
                    ).format(text, parent)
                )

            # Create the Figure entry in the DB
            paragraph = Paragraph(**parts)

            state["paragraph"]["idx"] += 1

            state["paragraph"]["text"] = text
            state["paragraph"]["field"] = field

            yield from self._parse_sentence(paragraph, node, state)

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

        # flattens children of node that are in the 'flatten' list
        if self.flatten:
            lxml.etree.strip_tags(root, self.flatten)
        # Assign the text, which was stripped of the 'flatten'-tags, to the document
        document.text = lxml.etree.tostring(root, encoding="unicode")

        # This dictionary contain the global state necessary to parse a
        # document and each context element. This reflects the relationships
        # defined in parser/models. This contains the state necessary to create
        # the respective Contexts within the document.
        state = {
            "visited": set(),
            "parent": {},  # map of parent[child] = node used to discover child
            "context": {},  # track the Context of each node (context['td'] = Cell)
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

        tokenized_sentences = []
        while stack:
            node = stack.pop()
            if node not in state["visited"]:
                state["visited"].add(node)  # mark as visited

                # Process
                if self.lingual:
                    tokenized_sentences += [y for y in self._parse_node(node, state)]
                else:
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

        if self.lingual:
            yield from self.enrich_tokenized_sentences_with_nlp(tokenized_sentences)
