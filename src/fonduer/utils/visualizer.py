import logging
import os
import subprocess
from builtins import object
from collections import defaultdict

from bs4 import BeautifulSoup
from IPython.display import display
from wand.color import Color
from wand.drawing import Drawing
from wand.image import Image

logger = logging.getLogger(__name__)


class Visualizer(object):
    """
    Object to display bounding boxes on a pdf document
    """

    def __init__(self, pdf_path):
        """
        :param pdf_path: directory where documents are stored
        :return:
        """
        self.pdf_path = pdf_path

    def display_boxes(self, pdf_file, boxes, alternate_colors=False):
        """
        Displays each of the bounding boxes passed in 'boxes' on images of the pdf
        pointed to by pdf_file
        boxes is a list of 5-tuples (page, top, left, bottom, right)
        """
        imgs = []
        colors = [Color("blue"), Color("red")]
        boxes_per_page = defaultdict(int)
        boxes_by_page = defaultdict(list)
        for i, (page, top, left, bottom, right) in enumerate(boxes):
            boxes_per_page[page] += 1
            boxes_by_page[page].append((top, left, bottom, right))
        for i, page_num in enumerate(boxes_per_page.keys()):
            img = pdf_to_img(pdf_file, page_num)
            draw = Drawing()
            draw.fill_color = Color("rgba(0, 0, 0, 0.0)")
            for j, (top, left, bottom, right) in enumerate(boxes_by_page[page_num]):
                draw.stroke_color = colors[j % 2] if alternate_colors else colors[0]
                draw.rectangle(left=left, top=top, right=right, bottom=bottom)
            draw(img)
            imgs.append(img)
        return imgs

    def display_candidates(self, candidates, pdf_file=None):
        """
        Displays the bounding boxes corresponding to candidates on an image of the pdf
        boxes is a list of 5-tuples (page, top, left, bottom, right)
        """
        if not pdf_file:
            pdf_file = os.path.join(
                self.pdf_path, candidates[0][0].span.sentence.document.name
            )
            if os.path.isfile(pdf_file + ".pdf"):
                pdf_file += ".pdf"
            elif os.path.isfile(pdf_file + ".PDF"):
                pdf_file += ".PDF"
            else:
                logger.error("display_candidates failed: pdf file missing.")
        boxes = [
            get_box(mention.span) for c in candidates for mention in c.get_mentions()
        ]
        imgs = self.display_boxes(pdf_file, boxes, alternate_colors=True)
        return display(*imgs)

    def display_words(self, sentences, target=None, pdf_file=None):
        if not pdf_file:
            pdf_file = os.path.join(self.pdf_path, sentences[0].document.name + ".pdf")
        boxes = []
        for sentence in sentences:
            for i, word in enumerate(sentence.words):
                if target is None or word == target:
                    boxes.append(
                        (
                            sentence.page[i],
                            sentence.top[i],
                            sentence.left[i],
                            sentence.bottom[i],
                            sentence.right[i],
                        )
                    )
        imgs = self.display_boxes(pdf_file, boxes)
        return display(*imgs)


def get_box(span):
    box = (
        min(span.get_attrib_tokens("page")),
        min(span.get_attrib_tokens("top")),
        max(span.get_attrib_tokens("left")),
        min(span.get_attrib_tokens("bottom")),
        max(span.get_attrib_tokens("right")),
    )
    return box


def get_pdf_dim(pdf_file):
    html_content = subprocess.check_output(
        "pdftotext -f {} -l {} -bbox '{}' -".format("1", "1", pdf_file), shell=True
    )
    soup = BeautifulSoup(html_content, "html.parser")
    pages = soup.find_all("page")
    page_width, page_height = (
        int(float(pages[0].get("width"))),
        int(float(pages[0].get("height"))),
    )
    return page_width, page_height


def pdf_to_img(pdf_file, page_num, pdf_dim=None):
    """
    Converts pdf file into image
    :param pdf_file: path to the pdf file
    :param page_num: page number to convert (index starting at 1)
    :return: wand image object
    """
    if not pdf_dim:
        pdf_dim = get_pdf_dim(pdf_file)
    page_width, page_height = pdf_dim
    img = Image(filename="{}[{}]".format(pdf_file, page_num - 1))
    img.resize(page_width, page_height)
    return img
