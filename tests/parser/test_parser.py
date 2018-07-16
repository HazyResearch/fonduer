#! /usr/bin/env python
"""
These tests expect that postgres is installed and that a database named
parser_test has been created for the purpose of testing.

If you are testing locally, you will need to create this db.
"""
import logging
import os

from fonduer import Meta
from fonduer.parser import OmniParser
from fonduer.parser.models import Document, Phrase
from fonduer.parser.parser import OmniParserUDF
from fonduer.parser.preprocessors import HTMLDocPreprocessor
from fonduer.parser.spacy_parser import Spacy

ATTRIBUTE = "parser_test"


def test_parse_md_details(caplog):
    """Unit test of the final results stored in the database of the md document.

    This test only looks at the final results such that the implementation of
    the OmniParserUDF's apply() can be modified.
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

    # Create an OmniParser and parse the md document
    omni = OmniParser(structural=True, lingual=True, visual=True, pdf_path=pdf_path)
    omni.apply(preprocessor, parallelism=PARALLEL)

    # Grab the md document
    doc = session.query(Document).order_by(Document.name).all()[0]
    assert doc.name == "md"

    # Check that doc has a figure
    assert len(doc.figures) == 1
    assert doc.figures[0].url == "http://placebear.com/200/200"
    assert doc.figures[0].position == 0
    assert doc.figures[0].stable_id == "md::figure:0"

    # Check that doc has a table
    assert len(doc.tables) == 1
    assert doc.tables[0].position == 0
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

    # Check that doc has phrases
    assert len(doc.phrases) == 45

    logger.info("Doc: {}".format(doc))
    for i, phrase in enumerate(doc.phrases):
        logger.info("    Phrase[{}]: {}".format(i, phrase.text))

    header = doc.phrases[0]
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
    assert header.ner_tags == ["O", "O"]
    assert header.dep_labels == ["compound", "ROOT"]


def test_simple_tokenizer(caplog):
    """Unit test of OmniParser on a single document with lingual features off."""
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

    omni = OmniParser(structural=True, lingual=False, visual=True, pdf_path=pdf_path)
    omni.apply(preprocessor, parallelism=PARALLEL)

    doc = session.query(Document).order_by(Document.name).all()[1]

    logger.info("Doc: {}".format(doc))
    for i, phrase in enumerate(doc.phrases):
        logger.info("    Phrase[{}]: {}".format(i, phrase.text))

    header = doc.phrases[0]
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

    assert len(doc.phrases) == 44


def test_parse_document_diseases(caplog):
    """Unit test of OmniParser on a single document.

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

    # Create an OmniParser and parse the diseases document
    omni = OmniParser(structural=True, lingual=True, visual=True, pdf_path=pdf_path)
    omni.apply(preprocessor, parallelism=PARALLEL)

    # Grab the diseases document
    doc = session.query(Document).order_by(Document.name).all()[0]
    assert doc.name == "diseases"

    logger.info("Doc: {}".format(doc))
    for phrase in doc.phrases:
        logger.info("    Phrase: {}".format(phrase.text))

    phrase = sorted(doc.phrases)[11]
    logger.info("  {}".format(phrase))
    # Test structural attributes
    assert phrase.xpath == "/html/body/table[1]/tbody/tr[3]/td[1]/p"
    assert phrase.html_tag == "p"
    assert phrase.html_attrs == ["class=s6", "style=padding-top: 1pt"]

    # Test visual attributes
    assert phrase.page == [1, 1, 1]
    assert phrase.top == [342, 296, 356]
    assert phrase.left == [318, 369, 318]

    # Test lingual attributes
    assert phrase.ner_tags == ["O", "O", "GPE"]
    assert phrase.dep_labels == ["ROOT", "prep", "pobj"]

    # 44 phrases expected in the "diseases" document.
    assert len(doc.phrases) == 36


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

    corpus_parser = OmniParser(
        structural=True, lingual=True, visual=False, pdf_path=pdf_path
    )
    corpus_parser.apply(doc_preprocessor, parallelism=PARALLEL)

    docs = session.query(Document).order_by(Document.name).all()

    for doc in docs:
        logger.info("Doc: {}".format(doc.name))
        for phrase in doc.phrases:
            logger.info("  Phrase: {}".format(phrase.text))

    assert session.query(Document).count() == 2
    assert session.query(Phrase).count() == 81


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

    # Create an OmniParser and parse the md document
    omni = OmniParser(structural=True, lingual=True, visual=True, pdf_path=pdf_path)
    omni.apply(preprocessor, parallelism=PARALLEL)

    # Grab the document
    doc = session.query(Document).order_by(Document.name).all()[0]

    # Grab the phrases parsed by the OmniParser
    phrases = list(doc.phrases)

    logger.warning("Doc: {}".format(doc))
    for i, phrase in enumerate(phrases):
        logger.warning("    Phrase[{}]: {}".format(i, phrase.html_attrs))

    # Phrases for testing
    sub_phrases = [
        {
            "index": 5,
            "attr": [
                "class=col-header",
                "hobbies=work:hard;play:harder",
                "type=phenotype",
                "style=background: #f1f1f1; color: aquamarine; font-size: 18px;",
            ],
        },
        {"index": 8, "attr": ["class=row-header", "style=background: #f1f1f1;"]},
        {"index": 10, "attr": ["class=cell", "style=text-align: center;"]},
    ]

    # Assertions
    assert all(phrases[p["index"]].html_attrs == p["attr"] for p in sub_phrases)
