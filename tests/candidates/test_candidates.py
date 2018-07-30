#! /usr/bin/env python
import logging
import os

from hardware_matchers import part_matcher, temp_matcher, volt_matcher
from hardware_spaces import MentionNgramsPart, MentionNgramsTemp, MentionNgramsVolt
from hardware_throttlers import temp_throttler, volt_throttler

from fonduer import (
    CandidateExtractor,
    Document,
    HTMLDocPreprocessor,
    Meta,
    Parser,
    Sentence,
    candidate_subclass,
    mention_subclass,
)

logger = logging.getLogger(__name__)
ATTRIBUTE = "stg_temp_max"
DB = "cand_test"


def test_cand_gen(caplog):
    """Run an end-to-end test on documents of the hardware domain."""
    caplog.set_level(logging.INFO)
    # SpaCy on mac has issue on parallel parseing
    if os.name == "posix":
        PARALLEL = 1
    else:
        PARALLEL = 2  # Travis only gives 2 cores

    max_docs = 10

    session = Meta.init("postgres://localhost:5432/" + DB).Session()

    docs_path = "tests/candidates/data/html/"
    pdf_path = "tests/candidates/data/pdf/"

    # Parsing
    doc_preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)
    corpus_parser = Parser(
        structural=True, lingual=True, visual=True, pdf_path=pdf_path
    )
    corpus_parser.apply(doc_preprocessor, parallelism=PARALLEL)
    num_docs = session.query(Document).count()
    logger.info("Docs: {}".format(num_docs))
    assert num_docs == max_docs
    num_sentences = session.query(Sentence).count()
    logger.info("Sentences: {}".format(num_sentences))

    # Mention Extraction
    part_ngrams = MentionNgramsPart(parts_by_doc=None, n_max=3)
    temp_ngrams = MentionNgramsTemp(n_max=2)
    volt_ngrams = MentionNgramsVolt(n_max=1)

    Part = mention_subclass("part", part_ngrams, part_matcher)
    Temp = mention_subclass("temp", temp_ngrams, temp_matcher)
    Volt = mention_subclass("volt", volt_ngrams, volt_matcher)

    assert session.query(Part).count() == 10

    # Candidate Extraction
    Part_Temp = candidate_subclass("Part_Temp", [Part, Temp])
    Part_Volt = candidate_subclass("Part_Volt", [Part, Volt])

    candidate_extractor = CandidateExtractor(
        [Part_Temp, Part_Volt], candidate_filter=[temp_throttler, volt_throttler]
    )

    docs = session.query(Document).order_by(Document.name).all()
    candidate_extractor.apply(docs, split=0, parallelism=PARALLEL)

    train_cands = session.query(Part_Temp).filter(Part_Temp.split == 0).all()
    logger.info("Number of candidates: {}".format(len(train_cands)))
