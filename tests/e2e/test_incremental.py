import logging
import os

import pytest
from snorkel.labeling import labeling_function

from fonduer import Meta
from fonduer.candidates import CandidateExtractor, MentionExtractor
from fonduer.candidates.models import Candidate
from fonduer.features import Featurizer
from fonduer.features.models import Feature, FeatureKey
from fonduer.parser import Parser
from fonduer.parser.models import Document
from fonduer.parser.preprocessors import HTMLDocPreprocessor
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
DB = "inc_test"
if "CI" in os.environ:
    CONN_STRING = (
        f"postgresql://{os.environ['PGUSER']}:{os.environ['PGPASSWORD']}"
        + f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{DB}"
    )
else:
    CONN_STRING = f"postgresql://127.0.0.1:5432/{DB}"


@pytest.mark.skipif(
    "CI" not in os.environ, reason="Only run incremental on GitHub Actions"
)
def test_incremental():
    """Run an end-to-end test on incremental additions."""
    # GitHub Actions gives 2 cores
    # help.github.com/en/actions/reference/virtual-environments-for-github-hosted-runners
    PARALLEL = 2

    max_docs = 1

    session = Meta.init(CONN_STRING).Session()

    docs_path = "tests/data/html/dtc114w.html"
    pdf_path = "tests/data/pdf/dtc114w.pdf"

    doc_preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)

    corpus_parser = Parser(
        session,
        parallelism=PARALLEL,
        structural=True,
        lingual=True,
        visual=True,
        pdf_path=pdf_path,
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
    assert session.query(Feature).count() == 70
    assert session.query(FeatureKey).count() == 512

    F_train = featurizer.get_feature_matrices(train_cands)
    assert F_train[0].shape == (70, 512)
    assert len(featurizer.get_keys()) == 512

    # Test Dropping FeatureKey
    featurizer.drop_keys(["CORE_e1_LENGTH_1"])
    assert session.query(FeatureKey).count() == 512

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
    assert session.query(Label).count() == 70

    # Only 5 because LF_operating_row doesn't apply to the first test doc
    assert session.query(LabelKey).count() == 5
    L_train = labeler.get_label_matrices(train_cands)
    assert L_train[0].shape == (70, 5)
    assert len(labeler.get_keys()) == 5

    docs_path = "tests/data/html/112823.html"
    pdf_path = "tests/data/pdf/112823.pdf"

    doc_preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)

    corpus_parser.apply(doc_preprocessor, pdf_path=pdf_path, clear=False)

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
    assert len(train_cands[0]) == 1502

    # Test if existing candidates are skipped.
    candidate_extractor.apply(new_docs, split=0, parallelism=PARALLEL, clear=False)
    train_cands = candidate_extractor.get_candidates(split=0)
    assert len(train_cands) == 1
    assert len(train_cands[0]) == 1502

    # Update features
    featurizer.update(new_docs, parallelism=PARALLEL)
    assert session.query(Feature).count() == 1502
    assert session.query(FeatureKey).count() == 2573
    F_train = featurizer.get_feature_matrices(train_cands)
    assert F_train[0].shape == (1502, 2573)
    assert len(featurizer.get_keys()) == 2573

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
    assert session.query(Label).count() == 1502
    # Only 5 because LF_storage_row doesn't apply to any doc (always ABSTAIN)
    assert session.query(LabelKey).count() == 5
    L_train = labeler.get_label_matrices(train_cands)
    assert L_train[0].shape == (1502, 5)

    # Test clear
    featurizer.clear(train=True)
    assert session.query(FeatureKey).count() == 0
