#! /usr/bin/env python
import logging
import os

import pytest

from fonduer import (
    CandidateExtractor,
    Document,
    HTMLDocPreprocessor,
    MentionExtractor,
    Meta,
    Parser,
    Sentence,
    candidate_subclass,
    mention_subclass,
)
from fonduer.candidates.models import Candidate
from tests.shared.hardware_matchers import part_matcher, temp_matcher, volt_matcher
from tests.shared.hardware_spaces import (
    MentionNgramsPart,
    MentionNgramsTemp,
    MentionNgramsVolt,
)
from tests.shared.hardware_throttlers import temp_throttler, volt_throttler

logger = logging.getLogger(__name__)
ATTRIBUTE = "stg_temp_max"
DB = "cand_test"


def test_span_char_start_and_char_end(caplog):
    """Test extracting candidates from mentions from documents."""
    caplog.set_level(logging.INFO)
    PARALLEL = 1
    max_docs = 1
    session = Meta.init("postgres://localhost:5432/" + DB).Session()

    docs_path = "tests/data/html/112823.html"
    pdf_path = "tests/data/pdf/112823.pdf"

    # Parsing
    logger.info("Parsing...")
    doc_preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)
    corpus_parser = Parser(
        structural=True, lingual=True, visual=True, pdf_path=pdf_path
    )
    corpus_parser.apply(doc_preprocessor, parallelism=PARALLEL)
    assert session.query(Document).count() == max_docs
    assert session.query(Sentence).count() == 828
    docs = session.query(Document).order_by(Document.name).all()

    # Mention Extraction
    part_ngrams = MentionNgramsPart(parts_by_doc=None, n_max=3)

    Part = mention_subclass("Part")

    mention_extractor = MentionExtractor([Part], [part_ngrams], [part_matcher])
    mention_extractor.apply(docs, parallelism=PARALLEL)

    assert session.query(Part).count() == 70
    parts = [x for x in session.query(Part).all() if x.span.get_span() == "BC548BG"]
    assert len(parts) == 1
    part = parts[0]

    logger.info("Part: {}".format(part.span))

    assert part.span.get_span() == "BC548BG"
    assert part.span.char_start == 0
    assert part.span.char_end == 6


def test_cand_gen(caplog):
    """Test extracting candidates from mentions from documents."""
    caplog.set_level(logging.INFO)
    # SpaCy on mac has issue on parallel parseing
    if os.name == "posix":
        PARALLEL = 1
    else:
        PARALLEL = 2  # Travis only gives 2 cores

    max_docs = 10
    session = Meta.init("postgres://localhost:5432/" + DB).Session()

    docs_path = "tests/data/html/"
    pdf_path = "tests/data/pdf/"

    # Parsing
    num_docs = session.query(Document).count()
    if num_docs != max_docs:
        logger.info("Parsing...")
        doc_preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)
        corpus_parser = Parser(
            structural=True, lingual=True, visual=True, pdf_path=pdf_path
        )
        corpus_parser.apply(doc_preprocessor, parallelism=PARALLEL)
    assert session.query(Document).count() == max_docs
    assert session.query(Sentence).count() == 5892
    docs = session.query(Document).order_by(Document.name).all()

    # Mention Extraction
    part_ngrams = MentionNgramsPart(parts_by_doc=None, n_max=3)
    temp_ngrams = MentionNgramsTemp(n_max=2)
    volt_ngrams = MentionNgramsVolt(n_max=1)

    Part = mention_subclass("Part")
    Temp = mention_subclass("Temp")
    Volt = mention_subclass("Volt")

    with pytest.raises(ValueError):
        mention_extractor = MentionExtractor(
            [Part, Temp, Volt],
            [part_ngrams, volt_ngrams],  # Fail, mismatched arity
            [part_matcher, temp_matcher, volt_matcher],
        )
    with pytest.raises(ValueError):
        mention_extractor = MentionExtractor(
            [Part, Temp, Volt],
            [part_ngrams, temp_matcher, volt_ngrams],
            [part_matcher, temp_matcher],  # Fail, mismatched arity
        )

    mention_extractor = MentionExtractor(
        [Part, Temp, Volt],
        [part_ngrams, temp_ngrams, volt_ngrams],
        [part_matcher, temp_matcher, volt_matcher],
    )
    mention_extractor.apply(docs, parallelism=PARALLEL)

    assert session.query(Part).count() == 234
    assert session.query(Volt).count() == 108
    assert session.query(Temp).count() == 118
    part = session.query(Part).order_by(Part.id).all()[0]
    volt = session.query(Volt).order_by(Volt.id).all()[0]
    temp = session.query(Temp).order_by(Temp.id).all()[0]
    logger.info("Part: {}".format(part.span))
    logger.info("Volt: {}".format(volt.span))
    logger.info("Temp: {}".format(temp.span))

    # Candidate Extraction
    PartTemp = candidate_subclass("PartTemp", [Part, Temp])
    PartVolt = candidate_subclass("PartVolt", [Part, Volt])

    candidate_extractor = CandidateExtractor(
        [PartTemp, PartVolt], throttlers=[temp_throttler, volt_throttler]
    )

    candidate_extractor.apply(docs, split=0, parallelism=PARALLEL)

    assert session.query(PartTemp).count() == 3385
    assert session.query(PartVolt).count() == 3364
    assert session.query(Candidate).count() == 6749
    assert docs[0].name == "112823"
    assert len(docs[0].parts) == 70
    assert len(docs[0].volts) == 33
    assert len(docs[0].temps) == 18

    # Test that deletion of a Candidate does not delete the Mention
    session.query(PartTemp).delete()
    assert session.query(PartTemp).count() == 0
    assert session.query(Temp).count() == 118
    assert session.query(Part).count() == 234

    # Test deletion of Candidate if Mention is deleted
    assert session.query(PartVolt).count() == 3364
    assert session.query(Volt).count() == 108
    session.query(Volt).delete()
    assert session.query(Volt).count() == 0
    assert session.query(PartVolt).count() == 0
