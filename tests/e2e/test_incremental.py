"""Fonduer incremental e2e test."""
import logging
import os

import pytest
from snorkel.labeling import labeling_function

from fonduer.candidates import CandidateExtractor, MentionExtractor
from fonduer.candidates.models import Candidate
from fonduer.features import Featurizer
from fonduer.features.models import Feature, FeatureKey
from fonduer.parser import Parser
from fonduer.parser.models import Document
from fonduer.parser.preprocessors import HTMLDocPreprocessor
from fonduer.parser.visual_parser import PdfVisualParser
from fonduer.supervision import Labeler
from fonduer.supervision.labeler import ABSTAIN
from fonduer.supervision.models import Label, LabelKey
from tests.shared.hardware_lfs import (
    LF_negative_number_left,
    LF_operating_row,
    LF_storage_row,
    LF_temperature_row,
    LF_to_left,
    LF_tstg_row,
)
from tests.shared.hardware_matchers import part_matcher, temp_matcher
from tests.shared.hardware_spaces import MentionNgramsPart, MentionNgramsTemp
from tests.shared.hardware_subclasses import Part, PartTemp, Temp
from tests.shared.hardware_throttlers import temp_throttler

logger = logging.getLogger(__name__)
ATTRIBUTE = "stg_temp_max"


@pytest.mark.skipif(
    "CI" not in os.environ, reason="Only run incremental on GitHub Actions"
)
def test_incremental(database_session):
    """Run an end-to-end test on incremental additions."""
    # GitHub Actions gives 2 cores
    # help.github.com/en/actions/reference/virtual-environments-for-github-hosted-runners
    PARALLEL = 2

    max_docs = 1

    session = database_session

    docs_path = "tests/data/html/dtc114w.html"
    pdf_path = "tests/data/pdf/"

    doc_preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)

    corpus_parser = Parser(
        session,
        parallelism=PARALLEL,
        structural=True,
        lingual=True,
        visual=True,
        visual_parser=PdfVisualParser(pdf_path),
    )
    corpus_parser.apply(doc_preprocessor)

    num_docs = session.query(Document).count()
    logger.info(f"Docs: {num_docs}")
    assert num_docs == max_docs

    docs = corpus_parser.get_documents()
    last_docs = corpus_parser.get_documents()

    assert len(docs[0].sentences) == len(last_docs[0].sentences)

    # Mention Extraction
    part_ngrams = MentionNgramsPart(parts_by_doc=None, n_max=3)
    temp_ngrams = MentionNgramsTemp(n_max=2)

    mention_extractor = MentionExtractor(
        session, [Part, Temp], [part_ngrams, temp_ngrams], [part_matcher, temp_matcher]
    )

    mention_extractor.apply(docs, parallelism=PARALLEL)

    assert session.query(Part).count() == 11
    assert session.query(Temp).count() == 8

    # Test if clear=True works
    mention_extractor.apply(docs, parallelism=PARALLEL, clear=True)

    assert session.query(Part).count() == 11
    assert session.query(Temp).count() == 8

    # Candidate Extraction
    candidate_extractor = CandidateExtractor(
        session, [PartTemp], throttlers=[temp_throttler]
    )

    candidate_extractor.apply(docs, split=0, parallelism=PARALLEL)

    assert session.query(PartTemp).filter(PartTemp.split == 0).count() == 70
    assert session.query(Candidate).count() == 70

    # Grab candidate lists
    train_cands = candidate_extractor.get_candidates(split=0)
    assert len(train_cands) == 1
    assert len(train_cands[0]) == 70

    # Featurization
    featurizer = Featurizer(session, [PartTemp])

    featurizer.apply(split=0, train=True, parallelism=PARALLEL)
    assert session.query(Feature).count() == len(train_cands[0])
    num_feature_keys = session.query(FeatureKey).count()
    assert num_feature_keys == 514

    F_train = featurizer.get_feature_matrices(train_cands)
    assert F_train[0].shape == (len(train_cands[0]), num_feature_keys)
    assert len(featurizer.get_keys()) == num_feature_keys

    # Test Dropping FeatureKey
    featurizer.drop_keys(["BASIC_e1_LENGTH_1"])
    assert session.query(FeatureKey).count() == num_feature_keys - 1

    stg_temp_lfs = [
        LF_storage_row,
        LF_operating_row,
        LF_temperature_row,
        LF_tstg_row,
        LF_to_left,
        LF_negative_number_left,
    ]

    labeler = Labeler(session, [PartTemp])

    labeler.apply(split=0, lfs=[stg_temp_lfs], train=True, parallelism=PARALLEL)
    assert session.query(Label).count() == len(train_cands[0])

    # Only 5 because LF_operating_row doesn't apply to the first test doc
    assert session.query(LabelKey).count() == 5
    L_train = labeler.get_label_matrices(train_cands)
    assert L_train[0].shape == (len(train_cands[0]), 5)
    assert len(labeler.get_keys()) == 5

    docs_path = "tests/data/html/112823.html"
    pdf_path = "tests/data/pdf/"

    doc_preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)

    corpus_parser.apply(doc_preprocessor, clear=False)

    assert len(corpus_parser.get_documents()) == 2

    new_docs = corpus_parser.get_last_documents()

    assert len(new_docs) == 1
    assert new_docs[0].name == "112823"

    # Get mentions from just the new docs
    mention_extractor.apply(new_docs, parallelism=PARALLEL, clear=False)
    assert session.query(Part).count() == 81
    assert session.query(Temp).count() == 31

    # Test if existing mentions are skipped.
    mention_extractor.apply(new_docs, parallelism=PARALLEL, clear=False)
    assert session.query(Part).count() == 81
    assert session.query(Temp).count() == 31

    # Just run candidate extraction and assign to split 0
    candidate_extractor.apply(new_docs, split=0, parallelism=PARALLEL, clear=False)

    # Grab candidate lists
    train_cands = candidate_extractor.get_candidates(split=0)
    assert len(train_cands) == 1
    assert len(train_cands[0]) == 1501

    # Test if existing candidates are skipped.
    candidate_extractor.apply(new_docs, split=0, parallelism=PARALLEL, clear=False)
    train_cands = candidate_extractor.get_candidates(split=0)
    assert len(train_cands) == 1
    assert len(train_cands[0]) == 1501

    # Update features
    featurizer.update(new_docs, parallelism=PARALLEL)
    assert session.query(Feature).count() == len(train_cands[0])
    num_feature_keys = session.query(FeatureKey).count()
    assert num_feature_keys == 2608
    F_train = featurizer.get_feature_matrices(train_cands)
    assert F_train[0].shape == (len(train_cands[0]), num_feature_keys)
    assert len(featurizer.get_keys()) == num_feature_keys

    # Update LF_storage_row. Now it always returns ABSTAIN.
    @labeling_function(name="LF_storage_row")
    def LF_storage_row_updated(c):
        return ABSTAIN

    stg_temp_lfs = [
        LF_storage_row_updated,
        LF_operating_row,
        LF_temperature_row,
        LF_tstg_row,
        LF_to_left,
        LF_negative_number_left,
    ]

    # Update Labels
    labeler.update(docs, lfs=[stg_temp_lfs], parallelism=PARALLEL)
    labeler.update(new_docs, lfs=[stg_temp_lfs], parallelism=PARALLEL)
    assert session.query(Label).count() == len(train_cands[0])
    # Only 5 because LF_storage_row doesn't apply to any doc (always ABSTAIN)
    num_label_keys = session.query(LabelKey).count()
    assert num_label_keys == 5
    L_train = labeler.get_label_matrices(train_cands)
    assert L_train[0].shape == (len(train_cands[0]), num_label_keys)

    # Test clear
    featurizer.clear(train=True)
    assert session.query(FeatureKey).count() == 0
