"""Fonduer visual parser that parses visual information from PDF."""
import logging
import os
import re
import shutil
import subprocess
from builtins import range, zip
from collections import OrderedDict, defaultdict
from operator import attrgetter
from typing import DefaultDict, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from bs4 import BeautifulSoup
from bs4.element import Tag
from editdistance import eval as editdist  # Alternative library: python-levenshtein

from fonduer.parser.models import Sentence
from fonduer.parser.visual_parser.visual_parser import VisualParser
from fonduer.utils.utils_visual import Bbox

logger = logging.getLogger(__name__)

# Define a type alias for readability
# PdfWordId is an ID for a word that is unique within a PDF file.
# The two elements represent (page_num, i), where page_num is 1-based page number and
# i is a 0-based unique ID for a word within that page.
PdfWordId = Tuple[int, int]
# PdfWord is a type alias for (PdfWordId, str), where the second element is a string
# repsentation of the word.
PdfWord = Tuple[PdfWordId, str]
# Similarly, HtmlWordId is an ID for word that is unique within a HTML file.
# The two elements represent (sentence.stable_id, i), where i is an ID within a
# Sentence.
HtmlWordId = Tuple[str, int]
# Similar to PdfWord, HtmlWord is a type alias for (HtmlWordId, str), where the second
# element is a string repsentation of the word.
HtmlWord = Tuple[HtmlWordId, str]


class PdfVisualParser(VisualParser):
    """Link visual information, extracted from PDF, with parsed sentences.

    This linker assumes the following conditions for expected results:

    - The PDF file exists in a directory specified by `pdf_path`.
    - The basename of the PDF file is same as the *document name*
      and its extension is either ".pdf" or ".PDF".
    - A PDF has a text layer.
    """

    def __init__(self, pdf_path: str, verbose: bool = False) -> None:
        """Initialize VisualParser.

        :param pdf_path: a path to directory that contains PDF files.
        :param verbose: whether to turn on verbose logging.
        """
        if not os.path.isdir(pdf_path):
            raise ValueError(f"No directory exists at {pdf_path}!")
        self.pdf_path = pdf_path
        self.pdf_file: Optional[str] = None
        self.verbose = verbose
        self.coordinate_map: Optional[Dict[PdfWordId, Bbox]] = None
        self.pdf_word_list: Optional[List[PdfWord]] = None
        self.html_word_list: Optional[List[HtmlWord]] = None
        self.links: Optional[OrderedDict[HtmlWordId, PdfWordId]] = None
        self.pdf_dim: Optional[Tuple[int, int]] = None
        delimiters = (
            r"([\(\)\,\?\u2212\u201C\u201D\u2018\u2019\u00B0\*']|(?<!http):|\.$|\.\.\.)"
        )
        self.separators = re.compile(delimiters)

        # Check if poppler-utils is installed AND the version is 0.36.0 or above
        if shutil.which("pdfinfo") is None or shutil.which("pdftotext") is None:
            raise RuntimeError("poppler-utils is not installed or they are not in PATH")
        version = subprocess.check_output(
            "pdfinfo -v", shell=True, stderr=subprocess.STDOUT, universal_newlines=True
        )
        m = re.search(r"\d{1,2}\.\d{2}\.\d", version)
        if int(m.group(0).replace(".", "")) < 360:
            raise RuntimeError(
                f"Installed poppler-utils's version is {m.group(0)}, "
                f"but should be 0.36.0 or above"
            )

    def parse(
        self, document_name: str, sentences: Iterable[Sentence]
    ) -> Iterator[Sentence]:
        """Link visual information with sentences.

        :param document_name: the document name.
        :param sentences: sentences to be linked with visual information.
        :return: A generator of ``Sentence``.
        """
        # sentences should be sorted as their order is not deterministic.
        self.sentences = sorted(sentences, key=attrgetter("position"))
        self.pdf_file = self._get_linked_pdf_path(document_name)
        try:
            self._extract_pdf_words()
        except RuntimeError as e:
            logger.exception(e)
            return
        self._extract_html_words()
        self._link_lists(search_max=200)
        for sentence in self._update_coordinates():
            yield sentence

    def _extract_pdf_words(self) -> None:
        logger.debug(f"pdfinfo '{self.pdf_file}' | grep -a ^Pages: | sed 's/[^0-9]*//'")
        num_pages = subprocess.check_output(
            f"pdfinfo '{self.pdf_file}' | grep -a ^Pages: | sed 's/[^0-9]*//'",
            shell=True,
        )
        pdf_word_list: List[PdfWord] = []
        coordinate_map: Dict[PdfWordId, Bbox] = {}
        for i in range(1, int(num_pages) + 1):
            logger.debug(f"pdftotext -f {i} -l {i} -bbox-layout '{self.pdf_file}' -")
            html_content = subprocess.check_output(
                f"pdftotext -f {i} -l {i} -bbox-layout '{self.pdf_file}' -", shell=True
            )
            soup = BeautifulSoup(html_content, "html.parser")
            pages = soup.find_all("page")
            pdf_word_list_i, coordinate_map_i = self._coordinates_from_HTML(pages[0], i)
            pdf_word_list += pdf_word_list_i
            # update coordinate map
            coordinate_map.update(coordinate_map_i)
        self.pdf_word_list = pdf_word_list
        self.coordinate_map = coordinate_map
        if len(self.pdf_word_list) == 0:
            raise RuntimeError(
                f"Words could not be extracted from PDF: {self.pdf_file}"
            )
        # take last page dimensions
        page_width, page_height = (
            int(float(pages[0].get("width"))),
            int(float(pages[0].get("height"))),
        )
        self.pdf_dim = (page_width, page_height)
        if self.verbose:
            logger.info(f"Extracted {len(self.pdf_word_list)} pdf words")

    def _get_linked_pdf_path(self, document_name: str) -> str:
        """Get the pdf file path, return None if it doesn't exist.

        :param document_name: a document name.
        """
        full_path = os.path.join(self.pdf_path, document_name + ".pdf")
        if os.path.isfile(full_path):
            return full_path
        full_path = os.path.join(self.pdf_path, document_name + ".PDF")
        if os.path.isfile(full_path):
            return full_path

        return None

    def is_parsable(self, document_name: str) -> bool:
        """Verify that the file exists and has a PDF extension.

        :param document_name: The path to the PDF document.
        """
        return False if self._get_linked_pdf_path(document_name) is None else True

    def _coordinates_from_HTML(
        self, page: Tag, page_num: int
    ) -> Tuple[List[PdfWord], Dict[PdfWordId, Bbox]]:
        pdf_word_list: List[PdfWord] = []
        coordinate_map: Dict[PdfWordId, Bbox] = {}
        block_coordinates: Dict[PdfWordId, Tuple[int, int]] = {}
        blocks = page.find_all("block")
        i = 0  # counter for word_id in page_num
        for block in blocks:
            x_min_block = int(float(block.get("xmin")))
            y_min_block = int(float(block.get("ymin")))
            lines = block.find_all("line")
            for line in lines:
                y_min_line = int(float(line.get("ymin")))
                y_max_line = int(float(line.get("ymax")))
                words = line.find_all("word")
                for word in words:
                    xmin = int(float(word.get("xmin")))
                    xmax = int(float(word.get("xmax")))
                    for content in self.separators.split(word.getText()):
                        if len(content) > 0:  # Ignore empty characters
                            word_id: PdfWordId = (page_num, i)
                            pdf_word_list.append((word_id, content))
                            coordinate_map[word_id] = Bbox(
                                page_num,
                                y_min_line,
                                y_max_line,
                                xmin,
                                xmax,
                            )
                            block_coordinates[word_id] = (y_min_block, x_min_block)
                            i += 1
        # sort pdf_word_list by page, block top then block left, top, then left
        pdf_word_list = sorted(
            pdf_word_list,
            key=lambda word_id__: block_coordinates[word_id__[0]]
            + (coordinate_map[word_id__[0]].top, coordinate_map[word_id__[0]].left),
        )
        return pdf_word_list, coordinate_map

    def _extract_html_words(self) -> None:
        html_word_list: List[HtmlWord] = []
        for sentence in self.sentences:
            for i, word in enumerate(sentence.words):
                html_word_list.append(((sentence.stable_id, i), word))
        self.html_word_list = html_word_list
        if self.verbose:
            logger.info(f"Extracted {len(self.html_word_list)} html words")

    def _link_lists(
        self, search_max: int = 100, edit_cost: int = 20, offset_cost: int = 1
    ) -> None:
        # NOTE: there are probably some inefficiencies here from rehashing words
        # multiple times, but we're not going to worry about that for now

        def link_exact(l: int, u: int) -> None:
            l, u, L, U = get_anchors(l, u)
            # Inverted index that maps word to index(es) of html_word_list
            html_dict: DefaultDict[str, List[int]] = defaultdict(list)
            # Inverted index that maps word to index(es) of pdf_word_list
            pdf_dict: DefaultDict[str, List[int]] = defaultdict(list)
            for i, (_, word) in enumerate(self.html_word_list[l:u]):
                if html_to_pdf[l + i] is None:
                    html_dict[word].append(l + i)
            for j, (_, word) in enumerate(self.pdf_word_list[L:U]):
                if pdf_to_html[L + j] is None:
                    pdf_dict[word].append(L + j)
            for word, html_list in list(html_dict.items()):
                pdf_list = pdf_dict[word]
                if len(html_list) == len(pdf_list):
                    for k in range(len(html_list)):
                        html_to_pdf[html_list[k]] = pdf_list[k]
                        pdf_to_html[pdf_list[k]] = html_list[k]

        def link_fuzzy(i: int) -> None:
            (_, word) = self.html_word_list[i]
            l = u = i
            l, u, L, U = get_anchors(l, u)
            offset = int(L + float(i - l) / (u - l) * (U - L))
            searchIndices = np.clip(offset + search_order, 0, M - 1)
            cost = [0] * search_max
            for j, k in enumerate(searchIndices):
                other = self.pdf_word_list[k][1]
                if (
                    word.startswith(other)
                    or word.endswith(other)
                    or other.startswith(word)
                    or other.endswith(word)
                ):
                    html_to_pdf[i] = k
                    return
                else:
                    cost[j] = int(editdist(word, other)) * edit_cost + j * offset_cost
            html_to_pdf[i] = searchIndices[np.argmin(cost)]
            return

        def get_anchors(l: int, u: int) -> Tuple[int, int, int, int]:
            while l >= 0 and html_to_pdf[l] is None:
                l -= 1
            while u < N and html_to_pdf[u] is None:
                u += 1
            if l < 0:
                l = 0
                L = 0
            else:
                L = html_to_pdf[l]
            if u >= N:
                u = N
                U = M
            else:
                U = html_to_pdf[u]
            return l, u, L, U

        def display_match_counts() -> int:
            matches = sum(
                [
                    html_to_pdf[i] is not None
                    and self.html_word_list[i][1]
                    == self.pdf_word_list[html_to_pdf[i]][1]
                    for i in range(len(self.html_word_list))
                ]
            )
            total = len(self.html_word_list)
            logger.info(f"({matches}/{total}) = {matches / total:.2f}")
            return matches

        N = len(self.html_word_list)
        M = len(self.pdf_word_list)

        try:
            assert N > 0 and M > 0
        except Exception:
            logger.exception(f"N = {N} and M = {M} are invalid values.")

        html_to_pdf: List[Optional[int]] = [None] * N
        pdf_to_html: List[Optional[int]] = [None] * M
        search_radius = search_max // 2

        # first pass: global search for exact matches
        link_exact(0, N)
        if self.verbose:
            logger.debug("Global exact matching:")
            display_match_counts()

        # second pass: local search for exact matches
        for i in range(((N + 2) // search_radius) + 1):
            link_exact(
                max(0, i * search_radius - search_radius),
                min(N, i * search_radius + search_radius),
            )
        if self.verbose:
            logger.debug("Local exact matching:")
            display_match_counts()

        # third pass: local search for approximate matches
        search_order = np.array(
            [(-1) ** (i % 2) * (i // 2) for i in range(1, search_max + 1)]
        )
        for i in range(len(html_to_pdf)):
            if html_to_pdf[i] is None:
                link_fuzzy(i)
        if self.verbose:
            logger.debug("Local approximate matching:")
            display_match_counts()

        # convert list to dict
        matches = sum(
            [
                html_to_pdf[i] is not None
                and self.html_word_list[i][1] == self.pdf_word_list[html_to_pdf[i]][1]
                for i in range(len(self.html_word_list))
            ]
        )
        total = len(self.html_word_list)
        if self.verbose:
            logger.debug(
                f"Linked {matches}/{total} ({matches / total:.2f}) html words exactly"
            )
        self.links = OrderedDict(
            (self.html_word_list[i][0], self.pdf_word_list[html_to_pdf[i]][0])
            for i in range(len(self.html_word_list))
        )

    def _update_coordinates(self) -> Iterator[Sentence]:
        for sentence in self.sentences:
            (page, top, bottom, left, right) = list(
                zip(
                    *[
                        self.coordinate_map[self.links[((sentence.stable_id), i)]]
                        for i in range(len(sentence.words))
                    ]
                )
            )
            sentence.page = list(page)
            sentence.top = list(top)
            sentence.left = list(left)
            sentence.bottom = list(bottom)
            sentence.right = list(right)
            yield sentence
        if self.verbose:
            logger.debug("Updated coordinates in database")
