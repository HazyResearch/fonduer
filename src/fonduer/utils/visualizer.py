"""Fonduer visualizer."""
import logging
import os
import subprocess
import warnings
from builtins import object
from collections import defaultdict
from typing import DefaultDict, List, Optional, Tuple

from bs4 import BeautifulSoup
from IPython.display import DisplayHandle, display
from wand.color import Color
from wand.drawing import Drawing
from wand.image import Image

from fonduer.candidates.models import Candidate, SpanMention
from fonduer.parser.models import Sentence
from fonduer.utils.utils_visual import Bbox

logger = logging.getLogger(__name__)


class Visualizer(object):
    """Object to display bounding boxes on a pdf document."""

    def __init__(self, pdf_path: str) -> None:
        """Initialize Visualizer.

        :param pdf_path: directory where documents are stored
        :return:
        """
        self.pdf_path = pdf_path

    def display_boxes(
        self,
        pdf_file: str,
        boxes: List[Bbox],
        alternate_colors: bool = False,
    ) -> List[Image]:
        """Display bounding boxes on the document.

        Display each of the bounding boxes passed in 'boxes' on images of the pdf
        pointed to by pdf_file
        boxes is a list of 5-tuples (page, top, left, bottom, right)
        """
        imgs = []
        with Color("blue") as blue, Color("red") as red, Color(
            "rgba(0, 0, 0, 0.0)"
        ) as transparent:
            colors = [blue, red]
            boxes_per_page: DefaultDict[int, int] = defaultdict(int)
            boxes_by_page: DefaultDict[
                int, List[Tuple[int, int, int, int]]
            ] = defaultdict(list)
            for i, (page, top, bottom, left, right) in enumerate(boxes):
                boxes_per_page[page] += 1
                boxes_by_page[page].append((top, bottom, left, right))
            for i, page_num in enumerate(boxes_per_page.keys()):
                with Drawing() as draw:
                    img = pdf_to_img(pdf_file, page_num)
                    draw.fill_color = transparent
                    for j, (top, bottom, left, right) in enumerate(
                        boxes_by_page[page_num]
                    ):
                        draw.stroke_color = (
                            colors[j % 2] if alternate_colors else colors[0]
                        )
                        draw.rectangle(left=left, top=top, right=right, bottom=bottom)
                    draw(img)
                    imgs.append(img)
            return imgs

    def display_candidates(
        self, candidates: List[Candidate], pdf_file: Optional[str] = None
    ) -> DisplayHandle:
        """Display the bounding boxes of candidates.

        Display the bounding boxes corresponding to candidates on an image of the pdf
        boxes is a list of 5-tuples (page, top, left, bottom, right)
        """
        if not pdf_file:
            pdf_file = os.path.join(self.pdf_path, candidates[0].document.name)
            if os.path.isfile(pdf_file + ".pdf"):
                pdf_file += ".pdf"
            elif os.path.isfile(pdf_file + ".PDF"):
                pdf_file += ".PDF"
            else:
                logger.error("display_candidates failed: pdf file missing.")
        boxes = [m.context.get_bbox() for c in candidates for m in c.get_mentions()]
        imgs = self.display_boxes(pdf_file, boxes, alternate_colors=True)
        return display(*imgs)

    def display_words(
        self,
        sentences: List[Sentence],
        target: Optional[str] = None,
        pdf_file: Optional[str] = None,
    ) -> DisplayHandle:
        """Display the bounding boxes of words.

        Display the bounding boxes corresponding to words on the pdf.
        """
        if not pdf_file:
            pdf_file = os.path.join(self.pdf_path, sentences[0].document.name + ".pdf")
        boxes = []
        for sentence in sentences:
            for i, word in enumerate(sentence.words):
                if target is None or word == target:
                    boxes.append(
                        Bbox(
                            sentence.page[i],
                            sentence.top[i],
                            sentence.bottom[i],
                            sentence.left[i],
                            sentence.right[i],
                        )
                    )
        imgs = self.display_boxes(pdf_file, boxes)
        return display(*imgs)


def get_box(span: SpanMention) -> Bbox:
    """Get the bounding box."""
    warnings.warn(
        "get_box(span) is deprecated. Use span.get_bbox() instead.",
        DeprecationWarning,
    )
    return Bbox(
        min(span.get_attrib_tokens("page")),
        min(span.get_attrib_tokens("top")),
        max(span.get_attrib_tokens("bottom")),
        min(span.get_attrib_tokens("left")),
        max(span.get_attrib_tokens("right")),
    )


def get_pdf_dim(pdf_file: str, page: int = 1) -> Tuple[int, int]:
    """Get the dimension of a pdf.

    :param pdf_file: path to the pdf file
    :param page: page number (starting from 1) to get a dimension for
    :return: width, height
    """
    html_content = subprocess.check_output(
        f"pdftotext -f {page} -l {page} -bbox '{pdf_file}' -", shell=True
    )
    soup = BeautifulSoup(html_content, "html.parser")
    pages = soup.find_all("page")
    page_width, page_height = (
        int(float(pages[0].get("width"))),
        int(float(pages[0].get("height"))),
    )
    return page_width, page_height


def pdf_to_img(
    pdf_file: str, page_num: int, pdf_dim: Optional[Tuple[int, int]] = None
) -> Image:
    """Convert pdf file into image.

    :param pdf_file: path to the pdf file
    :param page_num: page number to convert (index starting at 1)
    :return: wand image object
    """
    if not pdf_dim:
        pdf_dim = get_pdf_dim(pdf_file)
    page_width, page_height = pdf_dim
    img = Image(filename=f"{pdf_file}[{page_num - 1}]")
    img.resize(page_width, page_height)
    return img
