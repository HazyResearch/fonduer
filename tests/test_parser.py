#!/usr/bin/env python
"""
These tests expect that postgres is installed and that a databased named
parser_test has been created for the purpose of testing.

If you are testing locally, you will need to create this db.
"""
import logging
import os
import pytest

@pytest.mark.skip(reason="Don't to install CoreNLP on Travis. Will be decprecated.")
def test_corenlp(caplog):
    """Run a simple parse using CoreNLP as our parser."""
    caplog.set_level(logging.INFO)
    logger = logging.getLogger()
    PARALLEL = 2  # Travis only gives 2 cores
    ATTRIBUTE = "parser_test"
    os.environ['FONDUERDBNAME'] = ATTRIBUTE
    os.environ[
        'SNORKELDB'] = 'postgres://localhost:5432/' + os.environ['FONDUERDBNAME']

    from fonduer import SnorkelSession

    session = SnorkelSession()

    from fonduer import HTMLPreprocessor, OmniParser

    docs_path = os.environ['FONDUERHOME'] + '/tests/data/html_simple/'
    pdf_path = os.environ['FONDUERHOME'] + '/tests/data/pdf_simple/'

    max_docs = 2
    doc_preprocessor = HTMLPreprocessor(docs_path, max_docs=max_docs)

    corpus_parser = OmniParser(
        structural=True, lingual=True, visual=False, pdf_path=pdf_path)
    corpus_parser.apply(doc_preprocessor, parallel=PARALLEL)

    from fonduer.models import Document, Phrase

    docs = session.query(Document).order_by(Document.name).all()

    for doc in docs:
        logger.info("Doc: {}".format(doc.name))
        for phrase in doc.phrases:
            logger.info("  Phrase: {}".format(phrase.text))

    assert session.query(Document).count() == 2
    assert session.query(Phrase).count() == 80


def test_spacy(caplog):
    """Run a simple parse using spaCy as our parser."""
    caplog.set_level(logging.INFO)
    logger = logging.getLogger()

    PARALLEL = 2  # Travis only gives 2 cores
    ATTRIBUTE = "parser_test"
    os.environ['FONDUERDBNAME'] = ATTRIBUTE
    os.environ[
        'SNORKELDB'] = 'postgres://localhost:5432/' + os.environ['FONDUERDBNAME']

    from fonduer import SnorkelSession

    session = SnorkelSession()

    from snorkel.parser.spacy_parser import Spacy
    from fonduer import HTMLPreprocessor, OmniParser

    docs_path = os.environ['FONDUERHOME'] + '/tests/data/html_simple/'
    pdf_path = os.environ['FONDUERHOME'] + '/tests/data/pdf_simple/'

    max_docs = 2
    doc_preprocessor = HTMLPreprocessor(docs_path, max_docs=max_docs)

    corpus_parser = OmniParser(
        structural=True,
        lingual=True,
        visual=False,
        pdf_path=pdf_path,
        lingual_parser=Spacy())
    corpus_parser.apply(doc_preprocessor, parallel=PARALLEL)

    from fonduer.models import Document, Phrase

    docs = session.query(Document).order_by(Document.name).all()

    for doc in docs:
        logger.info("Doc: {}".format(doc.name))
        for phrase in doc.phrases:
            logger.info("  Phrase: {}".format(phrase.text))

    assert session.query(Document).count() == 2
    assert session.query(Phrase).count() == 80
