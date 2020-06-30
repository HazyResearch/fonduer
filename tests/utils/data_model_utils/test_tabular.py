"""Fonduer data model's tabular utils' unit tests."""
import pytest

from fonduer.candidates import MentionNgrams
from fonduer.parser.preprocessors import HTMLDocPreprocessor
from fonduer.utils.data_model_utils.tabular import (
    get_max_col_num,
    get_min_col_num,
    is_tabular_aligned,
    same_cell,
    same_col,
    same_row,
    same_sentence,
)
from tests.parser.test_parser import get_parser_udf


@pytest.fixture()
def mention_setup():
    """Set up mentions."""
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

    # Create 1-gram span mentions
    space = MentionNgrams(n_min=1, n_max=1)
    mentions = [tc for tc in space.apply(doc)]
    return mentions


def test_same_row(mention_setup):
    """Test the parser with the md document."""
    mentions = mention_setup

    # Same row
    assert mentions[51].get_span() == "Joan"
    assert mentions[52].get_span() == "saag"
    assert same_row((mentions[51], mentions[52]))

    # Different row
    assert mentions[57].get_span() == "Sally"
    assert not same_row((mentions[51], mentions[57]))


def test_same_col(mention_setup):
    """Test the parser with the md document."""
    mentions = mention_setup

    # Different column
    assert mentions[51].get_span() == "Joan"
    assert mentions[52].get_span() == "saag"
    assert not same_col((mentions[51], mentions[52]))

    # Same column
    assert mentions[57].get_span() == "Sally"
    assert same_col((mentions[51], mentions[57]))


def test_is_tabular_aligned(mention_setup):
    """Test the parser with the md document."""
    mentions = mention_setup

    # tabular_aligned
    assert mentions[51].get_span() == "Joan"
    assert mentions[52].get_span() == "saag"
    assert is_tabular_aligned((mentions[51], mentions[52]))

    # not tabular_aligned
    assert mentions[58].get_span() == "vindaloo"
    assert not is_tabular_aligned((mentions[51], mentions[58]))


def test_same_cell(mention_setup):
    """Test the parser with the md document."""
    mentions = mention_setup

    # Different cell
    assert mentions[51].get_span() == "Joan"
    assert mentions[52].get_span() == "saag"
    assert not same_cell((mentions[51], mentions[52]))

    # Same cell
    assert mentions[53].get_span() == "paneer"
    assert same_cell((mentions[52], mentions[53]))


def test_same_sentence(mention_setup):
    """Test the parser with the md document."""
    mentions = mention_setup

    # Same sentence
    assert mentions[0].get_span() == "Sample"
    assert mentions[1].get_span() == "Markdown"
    assert same_sentence((mentions[0], mentions[1]))

    # Different sentence
    assert mentions[2].get_span() == "This"
    assert not same_sentence((mentions[0], mentions[2]))


def test_get_min_max_col_num(mention_setup):
    """Test the parser with the md document."""
    mentions = mention_setup

    # Non tabular mention
    assert mentions[0].get_span() == "Sample"
    assert not get_max_col_num(mentions[0])
    assert not get_min_col_num(mentions[0])

    # Tabular mention
    assert mentions[51].get_span() == "Joan"
    assert get_min_col_num(mentions[51]) == 0
    # TODO: it'd be better to use the mention that spans multiple cols
    assert get_max_col_num(mentions[51]) == 0