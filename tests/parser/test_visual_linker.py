"""Fonduer visual_parser unit tests."""
import random
from operator import attrgetter

import pytest
from bs4 import BeautifulSoup

from fonduer.parser.preprocessors import HTMLDocPreprocessor
from fonduer.parser.visual_parser import PdfVisualParser
from tests.parser.test_parser import get_parser_udf


def test_visual_parser_not_affected_by_order_of_sentences():
    """Test if visual_parser result is not affected by the order of sentences."""
    docs_path = "tests/data/html/2N6427.html"
    pdf_path = "tests/data/pdf/"

    # Initialize preprocessor, parser, visual_parser.
    # Note that parser is initialized with `visual=False` and that visual_parser
    # will be used to attach "visual" information to sentences after parsing.
    preprocessor = HTMLDocPreprocessor(docs_path)
    parser_udf = get_parser_udf(
        structural=True, lingual=False, tabular=True, visual=False
    )
    visual_parser = PdfVisualParser(pdf_path=pdf_path)

    doc = parser_udf.apply(next(preprocessor.__iter__()))
    # Sort sentences by sentence.position
    doc.sentences = sorted(doc.sentences, key=attrgetter("position"))
    sentences0 = [sent for sent in visual_parser.parse(doc.name, doc.sentences)]
    # Sort again in case visual_parser.link changes the order
    sentences0 = sorted(sentences0, key=attrgetter("position"))

    doc = parser_udf.apply(next(preprocessor.__iter__()))
    # Shuffle
    random.shuffle(doc.sentences)
    sentences1 = [sent for sent in visual_parser.parse(doc.name, doc.sentences)]
    # Sort sentences by sentence.position
    sentences1 = sorted(sentences1, key=attrgetter("position"))

    # This should hold as both sentences are sorted by their position
    assert all(
        [
            sent0.position == sent1.position
            for (sent0, sent1) in zip(sentences0, sentences1)
        ]
    )

    # The following assertion should hold if the visual_parser result is not affected
    # by the order of sentences.
    assert all(
        [sent0.left == sent1.left for (sent0, sent1) in zip(sentences0, sentences1)]
    )


def test_non_existent_pdf_path_should_fail():
    """Test if a non-existent raises an error."""
    pdf_path = "dummy_path"
    with pytest.raises(ValueError):
        PdfVisualParser(pdf_path=pdf_path)


def test_pdf_word_list_is_sorted():
    """Test if pdf_word_list is sorted as expected.

    no_image_unsorted.html is originally created from pdf_simple/no_image.pdf,
    but the order of html elements like block and word has been changed to see if
    pdf_word_list is sorted as expected.
    """
    docs_path = "tests/data/html_simple/no_image_unsorted.html"
    pdf_path = "tests/data/pdf_simple"
    visual_parser = PdfVisualParser(pdf_path=pdf_path)
    with open(docs_path) as f:
        soup = BeautifulSoup(f, "html.parser")
    page = soup.find_all("page")[0]
    pdf_word_list, coordinate_map = visual_parser._coordinates_from_HTML(page, 1)

    # Check if words are sorted by block top
    assert set([content for (_, content) in pdf_word_list[:2]]) == {"Sample", "HTML"}
    # Check if words are sorted by top
    assert [content for (_, content) in pdf_word_list[2:7]] == [
        "This",
        "is",
        "an",
        "html",
        "that",
    ]
    # Check if words are sorted by left (#449)
    assert [content for (_, content) in pdf_word_list[:2]] == ["Sample", "HTML"]
