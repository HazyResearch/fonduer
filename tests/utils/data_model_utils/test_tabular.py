"""Fonduer data model's tabular utils' unit tests."""
import pytest

from fonduer.candidates import MentionNgrams
from fonduer.parser.preprocessors import HTMLDocPreprocessor
from fonduer.utils.data_model_utils.tabular import (
    get_aligned_ngrams,
    get_cell_ngrams,
    get_col_ngrams,
    get_head_ngrams,
    get_max_col_num,
    get_max_row_num,
    get_min_col_num,
    get_min_row_num,
    get_neighbor_cell_ngrams,
    get_neighbor_sentence_ngrams,
    get_row_ngrams,
    get_sentence_ngrams,
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
    """Test the same_row function."""
    mentions = mention_setup

    # Same row
    assert mentions[51].get_span() == "Joan"
    assert mentions[52].get_span() == "saag"
    assert same_row((mentions[51], mentions[52]))

    # Different row
    assert mentions[57].get_span() == "Sally"
    assert not same_row((mentions[51], mentions[57]))


def test_same_col(mention_setup):
    """Test the same_col function."""
    mentions = mention_setup

    # Different column
    assert mentions[51].get_span() == "Joan"
    assert mentions[52].get_span() == "saag"
    assert not same_col((mentions[51], mentions[52]))

    # Same column
    assert mentions[57].get_span() == "Sally"
    assert same_col((mentions[51], mentions[57]))


def test_is_tabular_aligned(mention_setup):
    """Test the is_tabular_aligned function."""
    mentions = mention_setup

    # tabular_aligned
    assert mentions[51].get_span() == "Joan"
    assert mentions[52].get_span() == "saag"
    assert is_tabular_aligned((mentions[51], mentions[52]))

    # not tabular_aligned
    assert mentions[58].get_span() == "vindaloo"
    assert not is_tabular_aligned((mentions[51], mentions[58]))


def test_same_cell(mention_setup):
    """Test the same_cell function."""
    mentions = mention_setup

    # Different cell
    assert mentions[51].get_span() == "Joan"
    assert mentions[52].get_span() == "saag"
    assert not same_cell((mentions[51], mentions[52]))

    # Same cell
    assert mentions[53].get_span() == "paneer"
    assert same_cell((mentions[52], mentions[53]))


def test_same_sentence(mention_setup):
    """Test the same_sentence function."""
    mentions = mention_setup

    # Same sentence
    assert mentions[0].get_span() == "Sample"
    assert mentions[1].get_span() == "Markdown"
    assert same_sentence((mentions[0], mentions[1]))

    # Different sentence
    assert mentions[2].get_span() == "This"
    assert not same_sentence((mentions[0], mentions[2]))


def test_get_min_max_col_num(mention_setup):
    """Test the get_min_col_num and get_max_col_num function."""
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


def test_get_min_max_row_num(mention_setup):
    """Test the get_min_row_num and get_max_row_num function."""
    mentions = mention_setup

    # Non tabular mention
    assert mentions[0].get_span() == "Sample"
    assert not get_max_row_num(mentions[0])
    assert not get_min_row_num(mentions[0])

    # Tabular mention
    assert mentions[51].get_span() == "Joan"
    assert get_min_row_num(mentions[51]) == 1
    assert get_max_row_num(mentions[51]) == 1


def test_get_sentence_ngrams(mention_setup):
    """Test the get_sentence_ngrams function."""
    mentions = mention_setup

    assert mentions[5].get_span() == "basic"
    assert list(get_sentence_ngrams(mentions[5])) == [
        "this",
        "is",
        "some",
        ",",
        "sample",
        "markdown",
        ".",
    ]


def test_get_neighbor_sentence_ngrams(mention_setup):
    """Test the get_neighbor_sentence_ngrams function."""
    mentions = mention_setup

    assert mentions[5].get_span() == "basic"
    assert list(get_neighbor_sentence_ngrams(mentions[5])) == ["sample", "markdown"] + [
        "second",
        "heading",
    ]


def test_get_cell_ngrams(mention_setup):
    """Test the get_cell_ngrams function."""
    mentions = mention_setup

    assert mentions[52].get_span() == "saag"
    assert list(get_cell_ngrams(mentions[52])) == ["paneer"]

    # TODO: test get_cell_ngrams when there are other sentences in the cell.

    # when a mention is not tabular
    assert mentions[0].get_span() == "Sample"
    assert list(get_cell_ngrams(mentions[0])) == []


def test_get_neighbor_cell_ngrams(mention_setup):
    """Test the get_neighbor_cell_ngrams function."""
    mentions = mention_setup

    assert mentions[52].get_span() == "saag"
    # No directions
    assert list(get_neighbor_cell_ngrams(mentions[52])) == ["paneer"] + ["joan"] + [
        "medium"
    ] + ["lunch", "order"] + ["vindaloo"]

    # directions=True
    assert list(get_neighbor_cell_ngrams(mentions[52], directions=True)) == [
        "paneer",
        ("joan", "LEFT"),
        ("medium", "RIGHT"),
        ("lunch", "UP"),
        ("order", "UP"),
        ("vindaloo", "DOWN"),
    ]

    # when a mention is not tabular
    assert mentions[0].get_span() == "Sample"
    assert list(get_neighbor_cell_ngrams(mentions[0])) == ["markdown"]


def test_get_row_ngrams(mention_setup):
    """Test the get_row_ngrams function."""
    mentions = mention_setup

    assert mentions[52].get_span() == "saag"
    assert list(get_row_ngrams(mentions[52])) == ["paneer"] + ["joan"] + ["medium"] + [
        "$",
        "11",
    ]


def test_get_col_ngrams(mention_setup):
    """Test the get_col_ngrams function."""
    mentions = mention_setup

    assert mentions[52].get_span() == "saag"
    assert list(get_col_ngrams(mentions[52])) == ["paneer"] + ["lunch", "order"] + [
        "vindaloo"
    ] + ["lamb", "madras"]

    # when a mention is not tabular
    assert mentions[0].get_span() == "Sample"
    assert list(get_col_ngrams(mentions[0])) == []


def test_get_aligned_ngrams(mention_setup):
    """Test the get_aligned_ngrams function."""
    mentions = mention_setup

    assert mentions[52].get_span() == "saag"
    # TODO: ["paneer"] appears twice. Is this expected result?
    assert list(get_aligned_ngrams(mentions[52])) == ["paneer"] + ["joan"] + [
        "medium"
    ] + ["$", "11"] + ["paneer"] + ["lunch", "order"] + ["vindaloo"] + [
        "lamb",
        "madras",
    ]


def test_get_head_ngrams(mention_setup):
    """Test the get_head_ngrams function."""
    mentions = mention_setup

    assert mentions[52].get_span() == "saag"
    assert list(get_head_ngrams(mentions[52])) == ["joan"] + ["lunch", "order"]

    # when a mention is in the 1st column
    assert mentions[51].get_span() == "Joan"
    assert list(get_head_ngrams(mentions[51])) == []

    # when a mention is in the header row
    assert mentions[46].get_span() == "Name"
    assert list(get_head_ngrams(mentions[46])) == []

    # when a mention is not tabular
    assert mentions[0].get_span() == "Sample"
    assert list(get_head_ngrams(mentions[0])) == []
