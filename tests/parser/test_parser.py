#! /usr/bin/env python
"""
These tests expect that postgres is installed and that a database named
parser_test has been created for the purpose of testing.

If you are testing locally, you will need to create this db.
"""
import logging
import os

import pytest

from fonduer import Meta
from fonduer.parser import Parser
from fonduer.parser.models import Document, Sentence
from fonduer.parser.preprocessors import HTMLDocPreprocessor

ATTRIBUTE = "parser_test"
ATTRIBUTE = "parser_test"


def test_parse_md_details(caplog):
    """Unit test of the final results stored in the database of the md document.

    This test only looks at the final results such that the implementation of
    the ParserUDF's apply() can be modified.
    """
    caplog.set_level(logging.INFO)
    logger = logging.getLogger(__name__)
    session = Meta.init("postgres://localhost:5432/" + ATTRIBUTE).Session()

    PARALLEL = 1
    max_docs = 1
    docs_path = "tests/data/html_simple/md.html"
    pdf_path = "tests/data/pdf_simple/md.pdf"

    # Preprocessor for the Docs
    preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)

    # Create an Parser and parse the md document
    parser = Parser(
        structural=True,
        tabular=True,
        lingual=True,
        visual=True,
        pdf_path=pdf_path,
        language="en",
    )
    parser.apply(preprocessor, parallelism=PARALLEL)

    # Grab the md document
    doc = session.query(Document).order_by(Document.name).all()[0]
    assert doc.name == "md"

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

    logger.info("Doc: {}".format(doc))
    for i, sentence in enumerate(doc.sentences):
        logger.info("    Sentence[{}]: {}".format(i, sentence.text))

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


# TODO: Get 'alpha' language example, e.g. Japanese
def test_spacy_language_support(caplog):
    """Test for support of spacy (alpha) tokenization of languages other than English

    This test only looks at the final results such that the implementation of
    the ParserUDF's apply() can be modified.
    """
    caplog.set_level(logging.INFO)
    session = Meta.init("postgres://localhost:5432/" + ATTRIBUTE).Session()

    PARALLEL = 1
    max_docs = 1
    docs_path = "tests/data/parser_htmls/brot.html"
    # Preprocessor for the Docs
    preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)

    # Create an Parser and parse the md document
    parser = Parser(structural=True, tabular=True, lingual=True, language="de")
    parser.apply(preprocessor, parallelism=PARALLEL)
    # Grab the md doument
    doc = session.query(Document).order_by(Document.name).all()[0]
    assert doc.name == "brot"
    assert len(doc.sentences) == 841
    header = sorted(doc.sentences, key=lambda x: x.position)[0]
    # Confirm that alpha parser does perform NLP for non-alpha language
    # NOTE: These entities are technically incorrect, might be due bad spacy accuracy?
    assert header.ner_tags == ["O", "O", "O"]

    # TODO: Add unit tests for tonkenized foreign alpha language sentences
    docs_path = "tests/data/parser_htmls/brot.html"  # TODO: replace with japanese doc
    # Preprocessor for the Docs
    preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)
    parser = Parser(structural=True, tabular=True, lingual=False, language="ja")
    parser.apply(preprocessor, parallelism=PARALLEL)
    # Grab the md doument
    doc = session.query(Document).order_by(Document.name).all()[0]
    assert doc.name == "brot"
    assert len(doc.sentences) == 894


def test_warning_on_missing_pdf(caplog):
    """Test that a warning is issued on invalid pdf."""
    caplog.set_level(logging.INFO)
    session = Meta.init("postgres://localhost:5432/" + ATTRIBUTE).Session()

    PARALLEL = 1
    max_docs = 1
    docs_path = "tests/data/html_simple/md_para.html"
    pdf_path = "tests/data/pdf_simple/md_para_nonexistant.pdf"

    # Preprocessor for the Docs
    preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)

    # Create an Parser and parse the md document
    parser = Parser(
        structural=True, tabular=True, lingual=True, visual=True, pdf_path=pdf_path
    )
    with pytest.warns(RuntimeWarning):
        parser.apply(preprocessor, parallelism=PARALLEL)

    assert session.query(Document).count() == 1


def test_warning_on_incorrect_filename(caplog):
    """Test that a warning is issued on invalid pdf."""
    caplog.set_level(logging.INFO)
    session = Meta.init("postgres://localhost:5432/" + ATTRIBUTE).Session()

    PARALLEL = 1
    docs_path = "tests/data/html_simple/md_para.html"
    pdf_path = "tests/data/html_simple/md_para.html"

    # Preprocessor for the Docs
    preprocessor = HTMLDocPreprocessor(docs_path)

    # Create an Parser and parse the md document
    parser = Parser(
        structural=True, tabular=True, lingual=True, visual=True, pdf_path=pdf_path
    )
    with pytest.warns(RuntimeWarning):
        parser.apply(preprocessor, parallelism=PARALLEL)

    assert session.query(Document).count() == 1


def test_parse_md_paragraphs(caplog):
    """Unit test of Paragraph parsing."""
    caplog.set_level(logging.INFO)
    session = Meta.init("postgres://localhost:5432/" + ATTRIBUTE).Session()

    PARALLEL = 1
    max_docs = 1
    docs_path = "tests/data/html_simple/md_para.html"
    pdf_path = "tests/data/pdf_simple/md_para.pdf"

    # Preprocessor for the Docs
    preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)

    # Create an Parser and parse the md document
    parser = Parser(
        structural=True, tabular=True, lingual=True, visual=True, pdf_path=pdf_path
    )
    parser.apply(preprocessor, parallelism=PARALLEL)

    # Grab the document
    doc = session.query(Document).order_by(Document.name).all()[0]
    assert doc.name == "md_para"

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
    session = Meta.init("postgres://localhost:5432/" + ATTRIBUTE).Session()

    # SpaCy on mac has issue on parallel parseing
    if os.name == "posix":
        PARALLEL = 1
    else:
        PARALLEL = 2  # Travis only gives 2 cores

    max_docs = 2
    docs_path = "tests/data/html_simple/"
    pdf_path = "tests/data/pdf_simple/"

    # Preprocessor for the Docs
    preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)

    parser = Parser(structural=True, lingual=False, visual=True, pdf_path=pdf_path)
    parser.apply(preprocessor, parallelism=PARALLEL)

    doc = session.query(Document).order_by(Document.name).all()[1]

    logger.info("Doc: {}".format(doc))
    for i, sentence in enumerate(doc.sentences):
        logger.info("    Sentence[{}]: {}".format(i, sentence.text))

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


def test_parse_document_diseases(caplog):
    """Unit test of Parser on a single document.

    This tests both the structural and visual parse of the document.
    """
    caplog.set_level(logging.INFO)
    logger = logging.getLogger(__name__)
    session = Meta.init("postgres://localhost:5432/" + ATTRIBUTE).Session()

    # SpaCy on mac has issue on parallel parseing
    if os.name == "posix":
        PARALLEL = 1
    else:
        PARALLEL = 2  # Travis only gives 2 cores

    max_docs = 2
    docs_path = "tests/data/html_simple/"
    pdf_path = "tests/data/pdf_simple/"

    # Preprocessor for the Docs
    preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)

    # Create an Parser and parse the diseases document
    parser = Parser(structural=True, lingual=True, visual=True, pdf_path=pdf_path)
    parser.apply(preprocessor, parallelism=PARALLEL)

    # Grab the diseases document
    doc = session.query(Document).order_by(Document.name).all()[0]
    assert doc.name == "diseases"

    logger.info("Doc: {}".format(doc))
    for sentence in doc.sentences:
        logger.info("    Sentence: {}".format(sentence.text))

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
    logger.info("  {}".format(sentence))

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


def test_spacy_integration(caplog):
    """Run a simple e2e parse using spaCy as our parser.

    The point of this test is to actually use the DB just as would be
    done in a notebook by a user.
    """
    #  caplog.set_level(logging.INFO)
    logger = logging.getLogger(__name__)

    # SpaCy on mac has issue on parallel parseing
    if os.name == "posix":
        PARALLEL = 1
    else:
        PARALLEL = 2  # Travis only gives 2 cores

    session = Meta.init("postgres://localhost:5432/" + ATTRIBUTE).Session()

    docs_path = "tests/data/html_simple/"
    pdf_path = "tests/data/pdf_simple/"

    max_docs = 2
    doc_preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)

    corpus_parser = Parser(
        structural=True, lingual=True, visual=False, pdf_path=pdf_path
    )
    corpus_parser.apply(doc_preprocessor, parallelism=PARALLEL)

    docs = session.query(Document).order_by(Document.name).all()

    for doc in docs:
        logger.info("Doc: {}".format(doc.name))
        for sentence in doc.sentences:
            logger.info("  Sentence: {}".format(sentence.text))

    assert session.query(Document).count() == 2
    assert session.query(Sentence).count() == 82


def test_parse_style(caplog):
    """Test style tag parsing."""
    caplog.set_level(logging.INFO)
    logger = logging.getLogger(__name__)
    session = Meta.init("postgres://localhost:5432/" + ATTRIBUTE).Session()

    # SpaCy on mac has issue on parallel parseing
    if os.name == "posix":
        PARALLEL = 1
    else:
        PARALLEL = 2  # Travis only gives 2 cores

    max_docs = 1
    docs_path = "tests/data/html_extended/ext_diseases.html"
    pdf_path = "tests/data/pdf_extended/ext_diseases.pdf"

    # Preprocessor for the Docs
    preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)

    # Create an Parser and parse the md document
    parser = Parser(structural=True, lingual=True, visual=True, pdf_path=pdf_path)
    parser.apply(preprocessor, parallelism=PARALLEL)

    # Grab the document
    doc = session.query(Document).order_by(Document.name).all()[0]

    # Grab the sentences parsed by the Parser
    sentences = list(session.query(Sentence).order_by(Sentence.position).all())

    logger.warning("Doc: {}".format(doc))
    for i, sentence in enumerate(sentences):
        logger.warning("    Sentence[{}]: {}".format(i, sentence.html_attrs))

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
