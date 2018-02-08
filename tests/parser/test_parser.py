# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
These tests expect that postgres is installed and that a databased named
parser_test has been created for the purpose of testing.

If you are testing locally, you will need to create this db.
"""
import logging
import os
import pytest

ATTRIBUTE = "parser_test"
os.environ['FONDUERDBNAME'] = ATTRIBUTE
os.environ[
    'SNORKELDB'] = 'postgres://localhost:5432/' + os.environ['FONDUERDBNAME']

from fonduer import SnorkelSession
from fonduer import HTMLPreprocessor, OmniParser
from fonduer.models import Document, Phrase
from fonduer.parser import OmniParserUDF
from snorkel.parser import Spacy


@pytest.mark.skip(
    reason="Don't want to install CoreNLP on Travis. Will be deprecated.")
def test_corenlp(caplog):
    """Run a simple parse using CoreNLP as our parser."""
    caplog.set_level(logging.INFO)
    logger = logging.getLogger()
    PARALLEL = 2  # Travis only gives 2 cores

    session = SnorkelSession()

    docs_path = os.environ['FONDUERHOME'] + '/tests/data/html_simple/'
    pdf_path = os.environ['FONDUERHOME'] + '/tests/data/pdf_simple/'

    max_docs = 2
    doc_preprocessor = HTMLPreprocessor(docs_path, max_docs=max_docs)

    corpus_parser = OmniParser(
        structural=True, lingual=True, visual=False, pdf_path=pdf_path)
    corpus_parser.apply(doc_preprocessor, parallel=PARALLEL)

    docs = session.query(Document).order_by(Document.name).all()

    for doc in docs:
        logger.info("Doc: {}".format(doc.name))
        for phrase in doc.phrases:
            logger.info("  Phrase: {}".format(phrase.text))

    assert session.query(Document).count() == 2
    assert session.query(Phrase).count() == 80


def test_parse_structure(caplog):
    """Unit test of OmniParserUDF.parse_structure().

    This only tests the structural parse of the document.
    """
    logger = logging.getLogger()

    max_docs = 1
    docs_path = os.environ['FONDUERHOME'] + '/tests/data/html_simple/md.html'
    pdf_path = os.environ['FONDUERHOME'] + '/tests/data/pdf_simple/md.pdf'

    # Preprocessor for the Docs
    preprocessor = HTMLPreprocessor(docs_path, max_docs=max_docs)

    # Grab one document, text tuple from the preprocessor
    doc, text = next(preprocessor.generate())
    logger.info("    Text: {}".format(text))

    # Create an OmniParserUDF
    omni_udf = OmniParserUDF(
        True,  # structural
        ["style"],  # blacklist
        ["span", "br"],  # flatten
        '',  # flatten delim
        True,  # lingual
        True,  # strip
        [(u'[\u2010\u2011\u2012\u2013\u2014\u2212\uf02d]', '-')],  # replace
        True,  # tabular
        True,  # visual
        pdf_path,  # pdf path
        Spacy())  # lingual parser

    # Grab the phrases parsed by the OmniParser
    phrases = list(omni_udf.parse_structure(doc, text))

    logger.warning("Doc: {}".format(doc))
    for phrase in phrases:
        logger.warning("    Phrase: {}".format(phrase.text))

    header = phrases[0]
    # Test structural attributes
    assert header.xpath == '/html/body/h1'
    assert header.html_tag == 'h1'
    assert header.html_attrs == ['id=sample-markdown']

    # 44 phrases expected in the "md" document.
    assert len(phrases) == 44


def test_parse_document(caplog):
    """Unit test of OmniParser on a single document.

    This tests both the structural and visual parse of the document.
    """
    caplog.set_level(logging.INFO)
    logger = logging.getLogger()
    session = SnorkelSession()

    PARALLEL = 2
    max_docs = 2
    docs_path = os.environ['FONDUERHOME'] + '/tests/data/html_simple/'
    pdf_path = os.environ['FONDUERHOME'] + '/tests/data/pdf_simple/'

    # Preprocessor for the Docs
    preprocessor = HTMLPreprocessor(docs_path, max_docs=max_docs)

    # Create an OmniParser and parse the md document
    omni = OmniParser(
        structural=True, lingual=True, visual=True, pdf_path=pdf_path)
    omni.apply(preprocessor, parallel=PARALLEL)

    # Grab the phrases parsed by the OmniParser
    doc = session.query(Document).order_by(Document.name).all()[1]

    logger.info("Doc: {}".format(doc))
    for phrase in doc.phrases:
        logger.info("    Phrase: {}".format(phrase.text))

    header = doc.phrases[0]
    # Test structural attributes
    assert header.xpath == '/html/body/h1'
    assert header.html_tag == 'h1'
    assert header.html_attrs == ['id=sample-markdown']

    # Test visual attributes
    assert header.page == [1, 1]
    assert header.top == [35, 35]
    assert header.bottom == [61, 61]
    assert header.right == [111, 231]
    assert header.left == [35, 117]

    # Test lingual attributes
    assert header.ner_tags == ['ORG', 'ORG']
    assert header.dep_labels == ['compound', 'ROOT']

    # 44 phrases expected in the "md" document.
    assert len(doc.phrases) == 44


def test_spacy_integration(caplog):
    """Run a simple e2e parse using spaCy as our parser.

    The point of this test is to actually use the DB just as would be
    done in a notebook by a user.
    """
    caplog.set_level(logging.INFO)
    logger = logging.getLogger()

    PARALLEL = 2  # Travis only gives 2 cores

    session = SnorkelSession()

    docs_path = os.environ['FONDUERHOME'] + '/tests/data/html_simple/'
    pdf_path = os.environ['FONDUERHOME'] + '/tests/data/pdf_simple/'

    max_docs = 2
    doc_preprocessor = HTMLPreprocessor(docs_path, max_docs=max_docs)

    corpus_parser = OmniParser(
        structural=True, lingual=True, visual=False, pdf_path=pdf_path)
    corpus_parser.apply(doc_preprocessor, parallel=PARALLEL)

    docs = session.query(Document).order_by(Document.name).all()

    for doc in docs:
        logger.info("Doc: {}".format(doc.name))
        for phrase in doc.phrases:
            logger.info("  Phrase: {}".format(phrase.text))

    assert session.query(Document).count() == 2
    assert session.query(Phrase).count() == 80
