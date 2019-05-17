#! /usr/bin/env python
import logging
import os
from unittest.mock import patch

import pytest

from fonduer.parser.models import Document
from fonduer.parser.parser import ParserUDF
from fonduer.parser.preprocessors import (
    CSVDocPreprocessor,
    HTMLDocPreprocessor,
    TextDocPreprocessor,
    TSVDocPreprocessor,
)


def get_parser_udf(
    structural=True,  # structural information
    blacklist=["style", "script"],  # ignore tag types, default: style, script
    flatten=["span", "br"],  # flatten tag types, default: span, br
    language="en",
    lingual=True,  # lingual information
    strip=True,
    replacements=[("[\u2010\u2011\u2012\u2013\u2014\u2212]", "-")],
    tabular=True,  # tabular information
    visual=False,  # visual information
    pdf_path=None,
):
    """Return an instance of ParserUDF."""

    # Patch new_sessionmaker() under the namespace of fonduer.utils.udf
    # See more details in
    # https://docs.python.org/3/library/unittest.mock.html#where-to-patch
    with patch("fonduer.utils.udf.new_sessionmaker", autospec=True):
        parser_udf = ParserUDF(
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
    return parser_udf


def test_parse_md_details(caplog):
    """Test the parser with the md document."""
    caplog.set_level(logging.INFO)
    logger = logging.getLogger(__name__)

    docs_path = "tests/data/html_simple/md.html"
    pdf_path = "tests/data/pdf_simple/md.pdf"

    # Preprocessor for the Docs
    preprocessor = HTMLDocPreprocessor(docs_path)
    doc = next(preprocessor._parse_file(docs_path, "md"))

    # Check that doc has a name
    assert doc.name == "md"

    # Check that doc does not have any of these
    assert len(doc.figures) == 0
    assert len(doc.tables) == 0
    assert len(doc.cells) == 0
    assert len(doc.sentences) == 0

    # Create an Parser and parse the md document
    parser_udf = get_parser_udf(
        structural=True,
        tabular=True,
        lingual=True,
        visual=True,
        pdf_path=pdf_path,
        language="en",
    )
    for _ in parser_udf.apply(doc):
        pass

    # Check that doc has a figure
    assert len(doc.figures) == 1
    assert doc.figures[0].url == "http://placebear.com/200/200"
    assert doc.figures[0].position == 0
    assert doc.figures[0].section.position == 0
    assert doc.figures[0].stable_id == "md::figure:0"

    #  Check that doc has a table
    assert len(doc.tables) == 1
    assert doc.tables[0].position == 0
    assert doc.tables[0].section.position == 0
    assert doc.tables[0].document.name == "md"

    # Check that doc has cells
    assert len(doc.cells) == 16
    cells = list(doc.cells)
    assert cells[0].row_start == 0
    assert cells[0].col_start == 0
    assert cells[0].position == 0
    assert cells[0].document.name == "md"
    assert cells[0].table.position == 0

    assert cells[10].row_start == 2
    assert cells[10].col_start == 2
    assert cells[10].position == 10
    assert cells[10].document.name == "md"
    assert cells[10].table.position == 0

    # Check that doc has sentences
    assert len(doc.sentences) == 45
    sent = sorted(doc.sentences, key=lambda x: x.position)[25]
    assert sent.text == "Spicy"
    assert sent.table.position == 0
    assert sent.table.section.position == 0
    assert sent.cell.row_start == 0
    assert sent.cell.col_start == 2

    logger.info(f"Doc: {doc}")
    for i, sentence in enumerate(doc.sentences):
        logger.info(f"    Sentence[{i}]: {sentence.text}")

    header = sorted(doc.sentences, key=lambda x: x.position)[0]
    # Test structural attributes
    assert header.xpath == "/html/body/h1"
    assert header.html_tag == "h1"
    assert header.html_attrs == ["id=sample-markdown"]

    # Test visual attributes
    assert header.page == [1, 1]
    assert header.top == [35, 35]
    assert header.bottom == [61, 61]
    assert header.right == [111, 231]
    assert header.left == [35, 117]

    # Test lingual attributes
    assert header.ner_tags == ["ORG", "ORG"]
    assert header.dep_labels == ["compound", "ROOT"]

    # Test whether nlp information corresponds to sentence words
    for sent in doc.sentences:
        assert len(sent.words) == len(sent.lemmas)
        assert len(sent.words) == len(sent.pos_tags)
        assert len(sent.words) == len(sent.ner_tags)
        assert len(sent.words) == len(sent.dep_parents)
        assert len(sent.words) == len(sent.dep_labels)


@pytest.mark.skipif(
    "CI" not in os.environ, reason="Only run spacy non English test on Travis"
)
def test_spacy_german(caplog):
    """Test the parser with the md document."""
    caplog.set_level(logging.INFO)

    docs_path = "tests/data/pure_html/brot.html"

    # Preprocessor for the Docs
    preprocessor = HTMLDocPreprocessor(docs_path)
    doc = next(preprocessor._parse_file(docs_path, "md"))

    # Create an Parser and parse the md document
    parser_udf = get_parser_udf(
        structural=True, tabular=True, lingual=True, visual=False, language="de"
    )
    for _ in parser_udf.apply(doc):
        pass

    # Check that doc has sentences
    assert len(doc.sentences) == 841
    sent = sorted(doc.sentences, key=lambda x: x.position)[143]
    assert sent.ner_tags == [
        "O",
        "O",
        "LOC",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
    ]  # inaccurate
    assert sent.dep_labels == [
        "mo",
        "ROOT",
        "sb",
        "cm",
        "nk",
        "mo",
        "punct",
        "mo",
        "nk",
        "nk",
        "nk",
        "sb",
        "oc",
        "rc",
        "punct",
    ]


@pytest.mark.skipif(
    "CI" not in os.environ, reason="Only run spacy non English test on Travis"
)
def test_spacy_japanese(caplog):
    """Test the parser with the md document."""
    caplog.set_level(logging.INFO)

    # Test Japanese alpha tokenization
    docs_path = "tests/data/pure_html/japan.html"
    preprocessor = HTMLDocPreprocessor(docs_path)
    doc = next(preprocessor._parse_file(docs_path, "md"))
    parser_udf = get_parser_udf(
        structural=True, tabular=True, lingual=True, visual=False, language="ja"
    )
    for _ in parser_udf.apply(doc):
        pass

    assert len(doc.sentences) == 289
    sent = doc.sentences[42]
    assert sent.text == "当時マルコ・ポーロが辿り着いたと言われる"
    assert sent.words == ["当時", "マルコ", "・", "ポーロ", "が", "辿り着い", "た", "と", "言わ", "れる"]
    assert sent.pos_tags == [
        "NOUN",
        "PROPN",
        "SYM",
        "PROPN",
        "ADP",
        "VERB",
        "AUX",
        "ADP",
        "VERB",
        "AUX",
    ]
    assert sent.lemmas == [
        "当時",
        "マルコ-Marco",
        "・",
        "ポーロ-Polo",
        "が",
        "辿り着く",
        "た",
        "と",
        "言う",
        "れる",
    ]
    # Japanese sentences are only tokenized.
    assert sent.ner_tags == [""] * len(sent.words)
    assert sent.dep_labels == [""] * len(sent.words)


@pytest.mark.skipif(
    "CI" not in os.environ, reason="Only run spacy non English test on Travis"
)
def test_spacy_chinese(caplog):
    """Test the parser with the md document."""
    caplog.set_level(logging.INFO)

    # Test Chinese alpha tokenization
    docs_path = "tests/data/pure_html/chinese.html"
    preprocessor = HTMLDocPreprocessor(docs_path)
    doc = next(preprocessor._parse_file(docs_path, "md"))
    parser_udf = get_parser_udf(
        structural=True, tabular=True, lingual=True, visual=False, language="zh"
    )
    for _ in parser_udf.apply(doc):
        pass
    print(doc.sentences)
    assert len(doc.sentences) == 8
    sent = doc.sentences[1]
    assert sent.text == "我们和他对比谁更厉害!"
    assert sent.words == ["我们", "和", "他", "对比", "谁", "更", "厉害", "!"]
    # Chinese sentences are only tokenized.
    assert sent.ner_tags == ["", "", "", "", "", "", "", ""]
    assert sent.dep_labels == ["", "", "", "", "", "", "", ""]


def test_warning_on_missing_pdf(caplog):
    """Test that a warning is issued on invalid pdf."""
    caplog.set_level(logging.INFO)

    docs_path = "tests/data/html_simple/md_para.html"
    pdf_path = "tests/data/pdf_simple/md_para_nonexistant.pdf"

    # Preprocessor for the Docs
    preprocessor = HTMLDocPreprocessor(docs_path)
    doc = next(preprocessor._parse_file(docs_path, "md_para"))

    # Create an Parser and parse the md document
    parser_udf = get_parser_udf(
        structural=True, tabular=True, lingual=True, visual=True, pdf_path=pdf_path
    )
    with pytest.warns(RuntimeWarning) as record:
        for _ in parser_udf.apply(doc):
            pass
    assert len(record) == 1
    assert "Visual parse failed" in record[0].message.args[0]


def test_warning_on_incorrect_filename(caplog):
    """Test that a warning is issued on invalid pdf."""
    caplog.set_level(logging.INFO)

    docs_path = "tests/data/html_simple/md_para.html"
    pdf_path = "tests/data/html_simple/md_para.html"

    # Preprocessor for the Docs
    preprocessor = HTMLDocPreprocessor(docs_path)
    doc = next(preprocessor._parse_file(docs_path, "md_para"))

    # Create an Parser and parse the md document
    parser_udf = get_parser_udf(
        structural=True, tabular=True, lingual=True, visual=True, pdf_path=pdf_path
    )
    with pytest.warns(RuntimeWarning) as record:
        for _ in parser_udf.apply(doc):
            pass
    assert len(record) == 1
    assert "Visual parse failed" in record[0].message.args[0]


def test_parse_md_paragraphs(caplog):
    """Unit test of Paragraph parsing."""
    caplog.set_level(logging.INFO)

    docs_path = "tests/data/html_simple/md_para.html"
    pdf_path = "tests/data/pdf_simple/md_para.pdf"

    # Preprocessor for the Docs
    preprocessor = HTMLDocPreprocessor(docs_path)
    doc = next(preprocessor._parse_file(docs_path, "md_para"))

    # Check that doc has a name
    assert doc.name == "md_para"

    # Create an Parser and parse the md document
    parser_udf = get_parser_udf(
        structural=True, tabular=True, lingual=True, visual=True, pdf_path=pdf_path
    )
    for _ in parser_udf.apply(doc):
        pass

    # Check that doc has a figure
    assert len(doc.figures) == 6
    assert doc.figures[0].url == "http://placebear.com/200/200"
    assert doc.figures[0].position == 0
    assert doc.figures[0].section.position == 0
    assert len(doc.figures[0].captions) == 0
    assert doc.figures[0].stable_id == "md_para::figure:0"
    assert doc.figures[0].cell.position == 13
    assert (
        doc.figures[2].url
        == "http://html5doctor.com/wp-content/uploads/2010/03/kookaburra.jpg"
    )
    assert doc.figures[2].position == 2
    assert len(doc.figures[2].captions) == 1
    assert len(doc.figures[2].captions[0].paragraphs[0].sentences) == 3
    assert (
        doc.figures[2].captions[0].paragraphs[0].sentences[0].text
        == "Australian Birds."
    )
    assert len(doc.figures[4].captions) == 0
    assert (
        doc.figures[4].url
        == "http://html5doctor.com/wp-content/uploads/2010/03/pelican.jpg"
    )

    #  Check that doc has a table
    assert len(doc.tables) == 1
    assert doc.tables[0].position == 0
    assert doc.tables[0].section.position == 0

    # Check that doc has cells
    assert len(doc.cells) == 16
    cells = list(doc.cells)
    assert cells[0].row_start == 0
    assert cells[0].col_start == 0
    assert cells[0].position == 0
    assert cells[0].table.position == 0

    assert cells[10].row_start == 2
    assert cells[10].col_start == 2
    assert cells[10].position == 10
    assert cells[10].table.position == 0

    # Check that doc has sentences
    assert len(doc.sentences) == 51
    sentences = sorted(doc.sentences, key=lambda x: x.position)
    sent1 = sentences[1]
    sent2 = sentences[2]
    sent3 = sentences[3]
    assert sent1.text == "This is some basic, sample markdown."
    assert sent2.text == (
        "Unlike the other markdown document, however, "
        "this document actually contains paragraphs of text."
    )
    assert sent1.paragraph.position == 1
    assert sent1.section.position == 0
    assert sent2.paragraph.position == 1
    assert sent2.section.position == 0
    assert sent3.paragraph.position == 1
    assert sent3.section.position == 0

    assert len(doc.paragraphs) == 46
    assert len(doc.paragraphs[1].sentences) == 3
    assert len(doc.paragraphs[2].sentences) == 1


def test_simple_tokenizer(caplog):
    """Unit test of Parser on a single document with lingual features off."""
    caplog.set_level(logging.INFO)
    logger = logging.getLogger(__name__)

    docs_path = "tests/data/html_simple/md.html"
    pdf_path = "tests/data/pdf_simple/md.pdf"

    # Preprocessor for the Docs
    preprocessor = HTMLDocPreprocessor(docs_path)
    doc = next(preprocessor._parse_file(docs_path, "md"))

    # Check that doc has a name
    assert doc.name == "md"

    # Create an Parser and parse the md document
    parser_udf = get_parser_udf(
        structural=True, lingual=False, visual=True, pdf_path=pdf_path, language=None
    )
    for _ in parser_udf.apply(doc):
        pass

    logger.info(f"Doc: {doc}")
    for i, sentence in enumerate(doc.sentences):
        logger.info(f"    Sentence[{i}]: {sentence.text}")

    header = sorted(doc.sentences, key=lambda x: x.position)[0]
    # Test structural attributes
    assert header.xpath == "/html/body/h1"
    assert header.html_tag == "h1"
    assert header.html_attrs == ["id=sample-markdown"]

    # Test lingual attributes
    assert header.ner_tags == ["", ""]
    assert header.dep_labels == ["", ""]
    assert header.dep_parents == [0, 0]
    assert header.lemmas == ["", ""]
    assert header.pos_tags == ["", ""]

    assert len(doc.sentences) == 44


def test_parse_table_span(caplog):
    caplog.set_level(logging.INFO)
    logger = logging.getLogger(__name__)

    docs_path = "tests/data/html_simple/table_span.html"

    # Preprocessor for the Docs
    preprocessor = HTMLDocPreprocessor(docs_path)
    doc = next(preprocessor._parse_file(docs_path, "table_span"))

    # Check that doc has a name
    assert doc.name == "table_span"

    # Create an Parser and parse the document
    parser_udf = get_parser_udf(structural=True, lingual=True, visual=False)
    for _ in parser_udf.apply(doc):
        pass

    logger.info(f"Doc: {doc}")

    assert len(doc.sentences) == 1
    for sentence in doc.sentences:
        logger.info(f"    Sentence: {sentence.text}")


def test_parse_document_diseases(caplog):
    """Unit test of Parser on a single document.

    This tests both the structural and visual parse of the document.
    """
    caplog.set_level(logging.INFO)
    logger = logging.getLogger(__name__)

    docs_path = "tests/data/html_simple/diseases.html"
    pdf_path = "tests/data/pdf_simple/diseases.pdf"

    # Preprocessor for the Docs
    preprocessor = HTMLDocPreprocessor(docs_path)
    doc = next(preprocessor._parse_file(docs_path, "diseases"))

    # Check that doc has a name
    assert doc.name == "diseases"

    # Create an Parser and parse the diseases document
    parser_udf = get_parser_udf(
        structural=True, lingual=True, visual=True, pdf_path=pdf_path
    )
    for _ in parser_udf.apply(doc):
        pass

    logger.info(f"Doc: {doc}")
    for sentence in doc.sentences:
        logger.info(f"    Sentence: {sentence.text}")

    # Check captions
    assert len(doc.captions) == 2
    caption = sorted(doc.sentences, key=lambda x: x.position)[20]
    assert caption.paragraph.caption.position == 0
    assert caption.paragraph.caption.table.position == 0
    assert caption.text == "Table 1: Infectious diseases and where to find them."
    assert caption.paragraph.position == 18

    # Check figures
    assert len(doc.figures) == 0

    #  Check that doc has a table
    assert len(doc.tables) == 3
    assert doc.tables[0].position == 0
    assert doc.tables[0].document.name == "diseases"

    # Check that doc has cells
    assert len(doc.cells) == 25

    sentence = sorted(doc.sentences, key=lambda x: x.position)[10]
    logger.info(f"  {sentence}")

    # Check the sentence's cell
    assert sentence.table.position == 0
    assert sentence.cell.row_start == 2
    assert sentence.cell.col_start == 1
    assert sentence.cell.position == 4

    # Test structural attributes
    assert sentence.xpath == "/html/body/table[1]/tbody/tr[3]/td[1]/p"
    assert sentence.html_tag == "p"
    assert sentence.html_attrs == ["class=s6", "style=padding-top: 1pt"]

    # Test visual attributes
    assert sentence.page == [1, 1, 1]
    assert sentence.top == [342, 296, 356]
    assert sentence.left == [318, 369, 318]

    # Test lingual attributes
    assert sentence.ner_tags == ["O", "O", "GPE"]
    assert sentence.dep_labels == ["ROOT", "prep", "pobj"]

    assert len(doc.sentences) == 37


def test_parse_style(caplog):
    """Test style tag parsing."""
    caplog.set_level(logging.INFO)
    logger = logging.getLogger(__name__)

    docs_path = "tests/data/html_extended/ext_diseases.html"
    pdf_path = "tests/data/pdf_extended/ext_diseases.pdf"

    # Preprocessor for the Docs
    preprocessor = HTMLDocPreprocessor(docs_path)
    doc = next(preprocessor._parse_file(docs_path, "ext_diseases"))

    # Create an Parser and parse the diseases document
    parser_udf = get_parser_udf(
        structural=True, lingual=True, visual=True, pdf_path=pdf_path
    )
    for _ in parser_udf.apply(doc):
        pass

    # Grab the sentences parsed by the Parser
    sentences = doc.sentences

    logger.warning(f"Doc: {doc}")
    for i, sentence in enumerate(sentences):
        logger.warning(f"    Sentence[{i}]: {sentence.html_attrs}")

    # sentences for testing
    sub_sentences = [
        {
            "index": 6,
            "attr": [
                "class=col-header",
                "hobbies=work:hard;play:harder",
                "type=phenotype",
                "style=background: #f1f1f1; color: aquamarine; font-size: 18px;",
            ],
        },
        {"index": 9, "attr": ["class=row-header", "style=background: #f1f1f1;"]},
        {"index": 11, "attr": ["class=cell", "style=text-align: center;"]},
    ]

    # Assertions
    assert all(sentences[p["index"]].html_attrs == p["attr"] for p in sub_sentences)


def test_parse_error_doc_skipping(caplog):
    """Test skipping of faulty htmls."""
    caplog.set_level(logging.INFO)

    faulty_doc_path = "tests/data/html_faulty/ext_diseases_missing_table_tag.html"
    preprocessor = HTMLDocPreprocessor(faulty_doc_path)
    doc = next(
        preprocessor._parse_file(faulty_doc_path, "ext_diseases_missing_table_tag")
    )
    parser_udf = get_parser_udf(structural=True, lingual=True)
    sentence_lists = [x for x in parser_udf.apply(doc)]
    # No sentences are yielded for faulty document
    assert len(sentence_lists) == 0

    valid_doc_path = "tests/data/html_extended/ext_diseases.html"
    preprocessor = HTMLDocPreprocessor(valid_doc_path)
    doc = next(preprocessor._parse_file(valid_doc_path, "ext_diseases"))
    parser_udf = get_parser_udf(structural=True, lingual=True)
    sentence_lists = [x for x in parser_udf.apply(doc)]
    assert len(sentence_lists) == 37


def test_parse_multi_sections(caplog):
    """Test the parser with the radiology document."""
    caplog.set_level(logging.INFO)

    # Test multi-section html
    docs_path = "tests/data/pure_html/radiology.html"
    preprocessor = HTMLDocPreprocessor(docs_path)
    doc = next(preprocessor._parse_file(docs_path, "radiology"))
    parser_udf = get_parser_udf(
        structural=True, tabular=True, lingual=True, visual=False
    )
    for _ in parser_udf.apply(doc):
        pass

    assert len(doc.sections) == 5
    assert len(doc.paragraphs) == 30
    assert len(doc.sentences) == 35
    assert len(doc.figures) == 2

    assert doc.sections[0].name is None
    assert doc.sections[1].name == "label"
    assert doc.sections[2].name == "content"
    assert doc.sections[3].name == "image"

    assert doc.sections[2].paragraphs[0].name == "COMPARISON"
    assert doc.sections[2].paragraphs[1].name == "INDICATION"
    assert doc.sections[2].paragraphs[2].name == "FINDINGS"
    assert doc.sections[2].paragraphs[3].name == "IMPRESSION"


def test_text_doc_preprocessor(caplog):
    """Test ``TextDocPreprocessor`` with text document."""
    caplog.set_level(logging.INFO)

    # Test text document
    docs_path = "tests/data/various_format/text_format.txt"
    preprocessor = TextDocPreprocessor(docs_path)
    doc = next(preprocessor._parse_file(docs_path, "plain_text_format"))
    parser_udf = get_parser_udf(
        structural=True, tabular=True, lingual=True, visual=False
    )
    for _ in parser_udf.apply(doc):
        pass

    assert len(preprocessor) == 1
    assert len(doc.sections) == 1
    assert len(doc.paragraphs) == 1
    assert len(doc.sentences) == 57


def test_tsv_doc_preprocessor(caplog):
    """Test ``TSVDocPreprocessor`` with tsv document."""
    caplog.set_level(logging.INFO)

    # Test tsv document
    docs_path = "tests/data/various_format/tsv_format.tsv"
    preprocessor = TSVDocPreprocessor(docs_path, header=True)

    assert len(preprocessor) == 2

    preprocessor = TSVDocPreprocessor(docs_path, max_docs=1, header=True)
    doc = next(preprocessor._parse_file(docs_path, "tsv_format"))
    parser_udf = get_parser_udf(
        structural=True, tabular=True, lingual=True, visual=False
    )
    for _ in parser_udf.apply(doc):
        pass

    assert len(preprocessor) == 1
    assert doc.name == "9b28e780-ba48-4a53-8682-7c58c141a1b6"
    assert len(doc.sections) == 1
    assert len(doc.paragraphs) == 1
    assert len(doc.sentences) == 33


def test_csv_doc_preprocessor(caplog):
    """Test ``CSVDocPreprocessor`` with csv document."""
    caplog.set_level(logging.INFO)

    # Test csv document
    docs_path = "tests/data/various_format/csv_format.csv"
    preprocessor = CSVDocPreprocessor(docs_path, header=True)

    assert len(preprocessor) == 10

    preprocessor = CSVDocPreprocessor(docs_path, max_docs=1, header=True)
    doc = next(preprocessor._parse_file(docs_path, "csv_format"))
    parser_udf = get_parser_udf(
        structural=True, tabular=True, lingual=True, visual=False
    )
    for _ in parser_udf.apply(doc):
        pass

    assert len(preprocessor) == 1
    assert len(doc.sections) == 12
    assert len(doc.paragraphs) == 10
    assert len(doc.sentences) == 17


def test_parser_skips_and_flattens(caplog):
    """Test if ``Parser`` skips/flattens elements."""
    caplog.set_level(logging.INFO)

    parser_udf = get_parser_udf()

    # Test if a parser skips comments
    doc = Document(id=1, name="test", stable_id="1::document:0:0")
    doc.text = "<html><body>Hello!<!-- comment --></body></html>"
    for _ in parser_udf.apply(doc):
        pass
    assert doc.sentences[0].text == "Hello!"

    # Test if a parser skips blacklisted elements
    doc = Document(id=2, name="test2", stable_id="2::document:0:0")
    doc.text = "<html><body><script>alert('Hello');</script><p>Hello!</p></body></html>"
    for _ in parser_udf.apply(doc):
        pass
    assert doc.sentences[0].text == "Hello!"

    # Test if a parser flattens elements
    doc = Document(id=3, name="test3", stable_id="3::document:0:0")
    doc.text = "<html><body><span>Hello, <br>world!</span></body></html>"
    for _ in parser_udf.apply(doc):
        pass
    assert doc.sentences[0].text == "Hello, world!"

    # Now with different blacklist and flatten
    parser_udf = get_parser_udf(blacklist=["meta"], flatten=["word"])

    # Test if a parser does not skip non-blacklisted element
    doc = Document(id=4, name="test4", stable_id="4::document:0:0")
    doc.text = "<html><body><script>alert('Hello');</script><p>Hello!</p></body></html>"
    for _ in parser_udf.apply(doc):
        pass
    assert doc.sentences[0].text == "alert('Hello');"
    assert doc.sentences[1].text == "Hello!"

    # Test if a parser skips blacklisted elements
    doc = Document(id=5, name="test5", stable_id="5::document:0:0")
    doc.text = "<html><head><meta name='keywords'></head><body>Hello!</body></html>"
    for _ in parser_udf.apply(doc):
        pass
    assert doc.sentences[0].text == "Hello!"

    # Test if a parser does not flatten elements
    doc = Document(id=6, name="test6", stable_id="6::document:0:0")
    doc.text = "<html><body><span>Hello, <br>world!</span></body></html>"
    for _ in parser_udf.apply(doc):
        pass
    assert doc.sentences[0].text == "Hello,"
    assert doc.sentences[1].text == "world!"

    # Test if a parser flattens elements
    doc = Document(id=7, name="test7", stable_id="7::document:0:0")
    doc.text = "<html><body><word>Hello, </word><word>world!</word></body></html>"
    for _ in parser_udf.apply(doc):
        pass
    assert doc.sentences[0].text == "Hello, world!"
