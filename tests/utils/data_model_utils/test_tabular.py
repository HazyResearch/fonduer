"""Fonduer data model's tabular utils' unit tests."""
import pytest

from fonduer.candidates import MentionNgrams
from fonduer.parser.preprocessors import HTMLDocPreprocessor
from fonduer.utils.data_model_utils.tabular import same_cell, same_col, same_row
from tests.parser.test_parser import get_parser_udf


@pytest.fixture()
def doc_setup():
    """Set up document."""
    docs_path = "tests/data/html_simple/md.html"
    pdf_path = "tests/data/pdf_simple/md.pdf"

    # Preprocessor for the Docs
    preprocessor = HTMLDocPreprocessor(docs_path)
    doc = next(preprocessor.__iter__())

    # Create an Parser and parse the md document
    parser_udf = get_parser_udf(
        structural=True,
        tabular=True,
        lingual=True,
        visual=True,
        pdf_path=pdf_path,
        language="en",
    )
    doc = parser_udf.apply(doc)
    return doc


def test_same_row(doc_setup):
    """Test the parser with the md document."""
    doc = doc_setup

    # Create 1-gram span mentions
    space = MentionNgrams(n_min=1, n_max=1)
    mentions = [tc for tc in space.apply(doc)]

    # Same row
    assert mentions[51].get_span() == "Joan"
    assert mentions[52].get_span() == "saag"
    assert same_row((mentions[51], mentions[52]))

    # Different row
    assert mentions[57].get_span() == "Sally"
    assert not same_row((mentions[51], mentions[57]))


def test_same_col(doc_setup):
    """Test the parser with the md document."""
    doc = doc_setup

    # Create 1-gram span mentions
    space = MentionNgrams(n_min=1, n_max=1)
    mentions = [tc for tc in space.apply(doc)]

    # Different column
    assert mentions[51].get_span() == "Joan"
    assert mentions[52].get_span() == "saag"
    assert not same_col((mentions[51], mentions[52]))

    # Same column
    assert mentions[57].get_span() == "Sally"
    assert same_col((mentions[51], mentions[57]))


def test_same_cell(doc_setup):
    """Test the parser with the md document."""
    doc = doc_setup

    # Create 1-gram span mentions
    space = MentionNgrams(n_min=1, n_max=1)
    mentions = [tc for tc in space.apply(doc)]

    # Different cell
    assert mentions[51].get_span() == "Joan"
    assert mentions[52].get_span() == "saag"
    assert not same_cell((mentions[51], mentions[52]))

    # Same cell
    assert mentions[53].get_span() == "paneer"
    assert same_cell((mentions[52], mentions[53]))
