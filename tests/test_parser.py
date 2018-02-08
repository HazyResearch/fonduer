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
    """Unit test of parser.py:parse_structure.

    We do not need to touch the database for this unit test.
    """
    caplog.set_level(logging.INFO)
    logger = logging.getLogger()

    session = SnorkelSession()

    max_docs = 1
    docs_path = os.environ['FONDUERHOME'] + '/tests/data/html_simple/'
    pdf_path = os.environ['FONDUERHOME'] + '/tests/data/pdf_simple/'

    # Preprocessor for the Docs
    preprocessor = HTMLPreprocessor(docs_path, max_docs=max_docs)

    # Grab one document, text tuple from the preprocessor
    doc, text = next(preprocessor.generate())
    logger.info("Doc: {}".format(doc))
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
        False,  # visual
        pdf_path,  # pdf path
        Spacy())  # lingual parser

    # Grab the phrases parsed by the OmniParser
    omni_udf.parse_structure(doc, text)
    phrases = list(session.query(Phrase).all())

    # 44 phrases expected in the "md" document.
    assert len(phrases) == 44


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
