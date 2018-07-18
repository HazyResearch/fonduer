#! /usr/bin/env python
"""
These tests expect that postgres is installed and that a database named
visualizer_test has been created for the purpose of testing.

If you are testing locally, you will need to create this db.
"""
import logging
import sys

import pytest

from fonduer import (
    CandidateExtractor,
    Document,
    HTMLDocPreprocessor,
    Meta,
    OmniNgrams,
    OrganizationMatcher,
    Parser,
    candidate_subclass,
)
from fonduer.parser import Parser
from fonduer.parser.models import Document
from fonduer.parser.preprocessors import HTMLDocPreprocessor
from fonduer.utils.visualizer import *

ATTRIBUTE = "visualizer_test"


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Only run visualizer test on Linux"
)
def test_visualizer(caplog):
    """Unit test of visualizer using the md document.
    """
    caplog.set_level(logging.INFO)
    session = Meta.init("postgres://localhost:5432/" + ATTRIBUTE).Session()

    PARALLEL = 1
    max_docs = 1
    docs_path = "tests/data/html_simple/md.html"
    pdf_path = "tests/data/pdf_simple/md.pdf"

    # Preprocessor for the Docs
    preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)

    # Create an Parser and parse the md document
    omni = Parser(
        structural=True, tabular=True, lingual=True, visual=True, pdf_path=pdf_path
    )
    omni.apply(preprocessor, parallelism=PARALLEL)

    # Grab the md document
    doc = session.query(Document).order_by(Document.name).all()[0]
    assert doc.name == "md"

    Organization = candidate_subclass("Organization", ["organization"])
    organization_matcher = OrganizationMatcher()
    organization_ngrams = OmniNgrams(n_max=1)

    candidate_extractor = CandidateExtractor(
        Organization, [organization_ngrams], [organization_matcher]
    )
    candidate_extractor.apply([doc], split=0, parallelism=PARALLEL)

    cands = session.query(Organization).filter(Organization.split == 0).all()

    # Test visualizer
    vis = Visualizer(pdf_path)
    vis.display_candidates([cands[0]])
