"""Fonduer parser."""
import itertools
import logging
import re
import warnings
from builtins import range
from collections import defaultdict
from typing import (
    Any,
    Collection,
    Dict,
    Iterator,
    List,
    Optional,
    Pattern,
    Tuple,
    Union,
)

import lxml.etree
import lxml.html
from lxml.html import HtmlElement
from sqlalchemy.orm import Session

from fonduer.parser.lingual_parser import LingualParser, SimpleParser, SpacyParser
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
from fonduer.parser.visual_parser import VisualParser
from fonduer.utils.udf import UDF, UDFRunner

logger = logging.getLogger(__name__)


class Parser(UDFRunner):
    r"""Parses into documents into Fonduer's Data Model.

    :param session: The database session to use.
    :param parallelism: The number of processes to use in parallel. Default 1.
    :param structural: Whether to parse structural information from a DOM.
    :param blacklist: A list of tag types to ignore. Default ["style", "script"].
    :param flatten: A list of tag types to flatten. Default ["span", "br"]
    :param language: Which spaCy NLP language package. Default "en".
    :param lingual: Whether or not to include NLP information. Default True.
    :param lingual_parser: A custom lingual parser that inherits
        :class:`LingualParser <fonduer.parser.lingual_parser.LingualParser>`.
        When specified, `language` will be ignored.
        When not, :class:`Spacy` with `language` will be used.
    :param strip: Whether or not to strip whitespace during parsing. Default True.
    :param replacements: A list of tuples where the regex string in the
        first position is replaced by the character in the second position.
        Default [(u"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-")], which
        replaces various unicode variants of a hyphen (e.g. emdash, endash,
        minus, etc.) with a standard ASCII hyphen.
    :param tabular: Whether to include tabular information in the parse.
    :param visual_parser: A visual parser that parses visual information.
        Defaults to None (visual information is not parsed).
    """

    def __init__(
        self,
        session: Session,
        parallelism: int = 1,
        structural: bool = True,  # structural information
        blacklist: List[str] = [
            "style",
            "script",
        ],  # ignore tag types, default: style, script
        flatten: List[str] = ["span", "br"],  # flatten tag types, default: span, br
        language: str = "en",
        lingual: bool = True,  # lingual information
        lingual_parser: Optional[LingualParser] = None,
        strip: bool = True,
        replacements: List[Tuple[str, str]] = [
            ("[\u2010\u2011\u2012\u2013\u2014\u2212]", "-")
        ],
        tabular: bool = True,  # tabular information
        visual_parser: Optional[VisualParser] = None,  # visual parser
    ) -> None:
        """Initialize Parser."""
        super().__init__(
            session,
            ParserUDF,
            parallelism=parallelism,
            structural=structural,
            blacklist=blacklist,
            flatten=flatten,
            lingual=lingual,
            lingual_parser=lingual_parser,
            strip=strip,
            replacements=replacements,
            tabular=tabular,
            visual_parser=visual_parser,
            language=language,
        )

    def apply(  # type: ignore
        self,
        doc_loader: Collection[Document],
        clear: bool = True,
        parallelism: Optional[int] = None,
        progress_bar: bool = True,
    ) -> None:
        """Run the Parser.

        :param doc_loader: An iteratable of ``Documents`` to parse. Typically,
            one of Fonduer's document preprocessors.
        :param clear: Whether or not to clear the labels table before applying
            these LFs.
        :param parallelism: How many threads to use for extraction. This will
            override the parallelism value used to initialize the Labeler if
            it is provided.
        :param progress_bar: Whether or not to display a progress bar. The
            progress bar is measured per document.
        """
        super().apply(
            doc_loader,
            clear=clear,
            parallelism=parallelism,
            progress_bar=progress_bar,
        )

    def _add(self, session: Session, doc: Union[Document, None]) -> None:
        # Persist the object if no error happens during parsing.
        if doc:
            session.add(doc)
            session.commit()

    def clear(self) -> None:  # type: ignore
        """Clear all of the ``Context`` objects in the database."""
        self.session.query(Context).delete(synchronize_session="fetch")

    def get_last_documents(self) -> List[Document]:
        """Return the most recently successfully parsed list of ``Documents``.

        :return: A list of the most recently parsed ``Documents`` ordered by name.
        """
        return (
            self.session.query(Document)
            .filter(Document.name.in_(self.last_docs))
            .order_by(Document.name)
            .all()
        )

    def get_documents(self) -> List[Document]:
        """Return all the successfully parsed ``Documents`` in the database.

        :return: A list of all ``Documents`` in the database ordered by name.
        """
        # return (
        #     self.session.query(Document, Sentence)
        #     .join(Sentence, Document.id == Sentence.document_id)
        #     .all()
        # )
        # return self.session.query(Sentence).order_by(Sentence.name).all()
        return self.session.query(Document).order_by(Document.name).all()


class ParserUDF(UDF):
    """Parser UDF class."""

    def __init__(
        self,
        structural: bool,
        blacklist: Union[str, List[str]],
        flatten: Union[str, List[str]],
        lingual: bool,
        lingual_parser: Optional[LingualParser],
        strip: bool,
        replacements: List[Tuple[str, str]],
        tabular: bool,
        visual_parser: Optional[VisualParser],
        language: Optional[str],
        **kwargs: Any,
    ) -> None:
        """Initialize Parser UDF.

        :param replacements: a list of (_pattern_, _replace_) tuples where
            _pattern_ isinstance a regex and _replace_ is a character string.
            All occurents of _pattern_ in the text will be replaced by
            _replace_.
        """
        super().__init__(**kwargs)

        # structural (html) setup
        self.structural = structural
        self.blacklist = blacklist if isinstance(blacklist, list) else [blacklist]
        self.flatten = flatten if isinstance(flatten, list) else [flatten]

        # lingual setup
        self.language = language
        self.strip = strip
        self.replacements: List[Tuple[Pattern, str]] = []
        for (pattern, replace) in replacements:
            self.replacements.append((re.compile(pattern, flags=re.UNICODE), replace))

        self.lingual = lingual
        if lingual_parser:
            self.lingual_parser = lingual_parser
        else:
            self.lingual_parser = SpacyParser(self.language)
            # Fallback to SimpleParser if a tokenizer is not supported.
            if not self.lingual_parser.has_tokenizer_support():
                self.lingual_parser = SimpleParser()

        if self.lingual and not self.lingual_parser.has_NLP_support():
            logger.warning(
                f"Lingual mode will be turned off, "
                f"as spacy doesn't provide support for this "
                f"language ({self.language})"
            )
            self.lingual = False

        # tabular setup
        self.tabular = tabular

        # visual setup
        self.visual_parser = visual_parser

    def apply(  # type: ignore
        self, document: Document, **kwargs: Any
    ) -> Optional[Document]:
        """Parse a text in an instance of Document.

        :param document: document to parse.
        """
        try:
            [y for y in self.parse(document, document.text)]
            if self.visual_parser:
                if not self.visual_parser.is_parsable(document.name):
                    warnings.warn(
                        (
                            f"Visual parse failed. "
                            f"{document.name} not a PDF. "
                            f"Proceeding without visual parsing."
                        ),
                        RuntimeWarning,
                    )
                else:
                    # Add visual attributes
                    [
                        y
                        for y in self.visual_parser.parse(
                            document.name, document.sentences
                        )
                    ]
            return document
        except Exception as e:
            logging.exception(
                (
                    f"Document {document.name} not added to database, "
                    f"because of parse error: \n{e}"
                )
            )
            return None

    def _parse_table(self, node: HtmlElement, state: Dict[str, Any]) -> Dict[str, Any]:
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
            stable_id = f"{state['document'].name}::{'table'}:{state['table']['idx']}"

            # Set name for Table
            name = node.attrib["name"] if "name" in node.attrib else None

            # Create the Table in the DB
            parts = {}
            parts["document"] = state["document"]
            parts["stable_id"] = stable_id
            parts["name"] = name
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
                try:
                    row_end += int(node.get("rowspan")) - 1
                except ValueError:
                    logger.error(f"Rowspan has invalid value: '{node.get('rowspan')}'")

            col_end = col_start
            if "colspan" in node.attrib:
                try:
                    col_end += int(node.get("colspan")) - 1
                except ValueError:
                    logger.error(f"Colspan has invalid value: '{node.get('colspan')}'")

            # update grid with occupied cells
            for r, c in itertools.product(
                list(range(row_start, row_end + 1)), list(range(col_start, col_end + 1))
            ):
                state["table"][state["parent"][node].position]["grid"][(r, c)] = 1

            # Set name for Cell
            name = node.attrib["name"] if "name" in node.attrib else None

            # construct cell
            parts = defaultdict(list)
            parts["document"] = state["document"]
            parts["name"] = name
            parts["table"] = state["parent"][node]
            parts["row_start"] = row_start
            parts["row_end"] = row_end
            parts["col_start"] = col_start
            parts["col_end"] = col_end
            parts["position"] = state["table"][state["parent"][node].position][
                "cell_pos"
            ]
            stable_id = (
                f"{parts['document'].name}"
                f"::"
                f"{'cell'}"
                f":"
                f"{parts['table'].position}"
                f":"
                f"{row_start}"
                f":"
                f"{col_start}"
            )
            parts["stable_id"] = stable_id
            # Create the Cell in the DB
            state["context"][node] = Cell(**parts)

            # Update position
            state["table"][state["parent"][node].position]["col_idx"] += 1
            state["table"][state["parent"][node].position]["cell_pos"] += 1

        return state

    def _parse_figure(self, node: HtmlElement, state: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the figure node.

        :param node: The lxml img node to parse
        :param state: The global state necessary to place the node in context
            of the document as a whole.
        """
        if node.tag not in ["img", "figure"]:
            return state

        # Process the Figure
        stable_id = (
            f"{state['document'].name}"
            f"::"
            f"{'figure'}"
            f":"
            f"{state['figure']['idx']}"
        )

        # Set name for Figure
        name = node.attrib["name"] if "name" in node.attrib else None

        # img within a Figure get's processed in the parent Figure
        if node.tag == "img" and isinstance(state["parent"][node], Figure):
            return state

        # NOTE: We currently do NOT support nested figures.
        parts: Dict[str, Any] = {}
        parent = state["parent"][node]
        if isinstance(parent, Section):
            parts["section"] = parent
        elif isinstance(parent, Cell):
            parts["section"] = parent.table.section
            parts["cell"] = parent
        else:
            logger.warning(f"Figure is nested within {state['parent'][node]}")
            return state

        parts["document"] = state["document"]
        parts["stable_id"] = stable_id
        parts["name"] = name
        parts["position"] = state["figure"]["idx"]

        # If processing a raw img
        if node.tag == "img":
            # Create the Figure entry in the DB
            parts["url"] = node.get("src")
            state["context"][node] = Figure(**parts)
        elif node.tag == "figure":
            # Pull the image from a child img node, if one exists
            imgs = [child for child in node if child.tag == "img"]

            # In case the image from the child img node doesn't exist
            if len(imgs) == 0:
                logger.warning("No image found in Figure.")
                return state

            if len(imgs) > 1:
                logger.warning("Figure contains multiple images.")
                # Right now we don't support multiple URLs in the Figure context
                # As a workaround, just ignore the outer Figure and allow processing
                # of the individual images. We ignore the accompanying figcaption
                # by marking it as visited.
                for child in node:
                    if child.tag == "figcaption":
                        child.set("visited", "true")
                return state

            img = imgs[0]
            img.set("visited", "true")

            # Create the Figure entry in the DB
            parts["url"] = img.get("src")
            state["context"][node] = Figure(**parts)

        state["figure"]["idx"] += 1
        return state

    def _parse_sentence(
        self, paragraph: Paragraph, node: HtmlElement, state: Dict[str, Any]
    ) -> Iterator[Sentence]:
        """Parse the Sentences of the node.

        :param node: The lxml node to parse
        :param state: The global state necessary to place the node in context
            of the document as a whole.
        """
        text = state["paragraph"]["text"]
        field = state["paragraph"]["field"]

        # Set name for Sentence
        name = node.attrib["name"] if "name" in node.attrib else None

        # Lingual Parse
        document = state["document"]
        for parts in self.lingual_parser.split_sentences(text):
            abs_offset = state["sentence"]["abs_offset"]
            parts["abs_char_offsets"] = [
                char_offset + abs_offset for char_offset in parts["char_offsets"]
            ]
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
            parts["name"] = name
            state["sentence"]["abs_offset"] = abs_sentence_offset_end
            if self.structural:
                context_node = node.getparent() if field == "tail" else node
                tree = lxml.etree.ElementTree(state["root"])
                parts["xpath"] = tree.getpath(context_node)
                parts["html_tag"] = context_node.tag
                parts["html_attrs"] = [
                    "=".join(x)
                    for x in context_node.attrib.items()
                    if x[0] != "visited"
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
            parts["position"] = state["sentence"]["idx"]

            # If tabular, consider own Context first in case a Cell
            # was just created. Otherwise, defer to the parent.
            parent = paragraph
            if isinstance(parent, Paragraph):
                parts["section"] = parent.section
                parts["paragraph"] = parent
                if parent.cell:  # if True self.tabular is also always True
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

    def _parse_paragraph(
        self, node: HtmlElement, state: Dict[str, Any]
    ) -> Iterator[Sentence]:
        """Parse a Paragraph of the node.

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
        # Set name for Paragraph
        name = node.attrib["name"] if "name" in node.attrib else None

        if len(node.getchildren()) == 0:  # leaf node
            fields = ["text", "tail"]
        elif node.get("visited") == "text":  # .text was parsed already
            fields = ["tail"]
            node.set("visited", "true")
        else:
            fields = ["text"]
            node.set("visited", "text")
            self.stack.append(node)  # will visit again later for tail
        for field in fields:
            text = getattr(node, field)
            text = text.strip() if text and self.strip else text

            # Skip if "" or None
            if not text:
                continue

            # Run RegEx replacements
            for (rgx, replace) in self.replacements:
                text = rgx.sub(replace, text)

            # Process the Paragraph
            stable_id = (
                f"{state['document'].name}"
                f"::"
                f"{'paragraph'}"
                f":"
                f"{state['paragraph']['idx']}"
            )
            parts = {}
            parts["stable_id"] = stable_id
            parts["name"] = name
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
            elif isinstance(parent, Table):  # occurs with text in the tail of a table
                parts["section"] = parent.section
            else:
                raise NotImplementedError(
                    f"Para '{text}' parent must be Section, Caption, or Cell, "
                    f"not {parent}"
                )

            # Create the entry in the DB
            paragraph = Paragraph(**parts)

            state["paragraph"]["idx"] += 1

            state["paragraph"]["text"] = text
            state["paragraph"]["field"] = field

            yield from self._parse_sentence(paragraph, node, state)

    def _parse_section(
        self, node: HtmlElement, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse a Section of the node.

        Note that this implementation currently creates a Section at the
        beginning of the document and creates Section based on tag of node.

        :param node: The lxml node to parse
        :param state: The global state necessary to place the node in context
            of the document as a whole.
        """
        if node.tag not in ["html", "section"]:
            return state

        # Add a Section
        stable_id = (
            f"{state['document'].name}"
            f"::"
            f"{'section'}"
            f":"
            f"{state['section']['idx']}"
        )

        # Set name for Section
        name = node.attrib["name"] if "name" in node.attrib else None

        state["context"][node] = Section(
            document=state["document"],
            name=name,
            stable_id=stable_id,
            position=state["section"]["idx"],
        )
        state["section"]["idx"] += 1

        return state

    def _parse_caption(
        self, node: HtmlElement, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse a Caption of the node.

        :param node: The lxml node to parse
        :param state: The global state necessary to place the node in context
            of the document as a whole.
        """
        if node.tag not in ["caption", "figcaption"]:  # captions used in Tables
            return state

        # Add a Caption
        parent = state["parent"][node]
        stable_id = (
            f"{state['document'].name}"
            f"::"
            f"{'caption'}"
            f":"
            f"{state['caption']['idx']}"
        )

        # Set name for Section
        name = node.attrib["name"] if "name" in node.attrib else None

        if isinstance(parent, Table):
            state["context"][node] = Caption(
                document=state["document"],
                table=parent,
                figure=None,
                stable_id=stable_id,
                name=name,
                position=state["caption"]["idx"],
            )
        elif isinstance(parent, Figure):
            state["context"][node] = Caption(
                document=state["document"],
                table=None,
                figure=parent,
                stable_id=stable_id,
                name=name,
                position=state["caption"]["idx"],
            )
        else:
            raise NotImplementedError("Caption must be a child of Table or Figure.")
        state["caption"]["idx"] += 1

        return state

    def _parse_node(
        self, node: HtmlElement, state: Dict[str, Any]
    ) -> Iterator[Sentence]:
        """Entry point for parsing all node types.

        :param node: The lxml HTML node to parse
        :param state: The global state necessary to place the node in context
            of the document as a whole.
        :return: a *generator* of Sentences
        """
        # Processing on entry of node
        if node.get("visited") != "text":  # skip when .text has been parsed
            state = self._parse_section(node, state)

            state = self._parse_figure(node, state)

            if self.tabular:
                state = self._parse_table(node, state)

            state = self._parse_caption(node, state)

        yield from self._parse_paragraph(node, state)

    def parse(self, document: Document, text: str) -> Iterator[Sentence]:
        """Depth-first search over the provided tree.

        Implemented as an iterative procedure. The structure of the state
        needed to parse each node is also defined in this function.

        :param document: the Document context
        :param text: the structured text of the document (e.g. HTML)
        :return: a *generator* of Sentences.
        """
        self.stack = []

        root = lxml.html.fromstring(text)

        # flattens children of node that are in the 'flatten' list
        if self.flatten:
            lxml.etree.strip_tags(root, self.flatten)
        # Strip comments
        lxml.etree.strip_tags(root, lxml.etree.Comment)
        # Assign the text, which was stripped of the 'flatten'-tags, to the document
        document.text = lxml.etree.tostring(root, encoding="unicode")

        # This dictionary contain the global state necessary to parse a
        # document and each context element. This reflects the relationships
        # defined in parser/models. This contains the state necessary to create
        # the respective Contexts within the document.
        state = {
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
        self.stack.append(root)
        state["parent"][root] = document
        state["context"][root] = document

        tokenized_sentences: List[Sentence] = []
        while self.stack:
            node = self.stack.pop()
            if node.get("visited") != "true":

                # Process
                if self.lingual:
                    tokenized_sentences += [y for y in self._parse_node(node, state)]
                else:
                    yield from self._parse_node(node, state)

                # NOTE: This reversed() order is to ensure that the iterative
                # DFS matches the order that would be produced by a recursive
                # DFS implementation.
                if node.get("visited") != "true":
                    for child in reversed(node):
                        # Skip nodes that are blacklisted
                        if self.blacklist and child.tag in self.blacklist:
                            continue

                        self.stack.append(child)

                        # store the parent of the node, which is either the parent
                        # Context, or if the parent did not create a Context, then
                        # use the node's parent Context.
                        state["parent"][child] = (
                            state["context"][node]
                            if node in state["context"]
                            else state["parent"][node]
                        )
                else:
                    node.set("visited", "true")  # mark as visited

        if self.lingual:
            yield from self.lingual_parser.enrich_sentences_with_NLP(
                tokenized_sentences
            )
