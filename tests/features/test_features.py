#! /usr/bin/env python
import logging
from sys import platform

from fonduer import Meta
from fonduer.candidates import CandidateExtractor, MentionExtractor
from fonduer.candidates.models import candidate_subclass, mention_subclass
from fonduer.features import FeatureExtractor, Featurizer
from fonduer.features.models import Feature, FeatureKey
from fonduer.parser import Parser
from fonduer.parser.models import Document, Sentence
from fonduer.parser.preprocessors import HTMLDocPreprocessor
from tests.shared.hardware_matchers import part_matcher, temp_matcher
from tests.shared.hardware_spaces import MentionNgramsPart, MentionNgramsTemp

logger = logging.getLogger(__name__)
DB = "feature_test"


def test_feature_extraction(caplog):
    """Test extracting candidates from mentions from documents."""
    caplog.set_level(logging.INFO)

    if platform == "darwin":
        logger.info("Using single core.")
        PARALLEL = 1
    else:
        logger.info("Using two cores.")
        PARALLEL = 2  # Travis only gives 2 cores

    def do_nothing_matcher(fig):
        return True

    max_docs = 1
    session = Meta.init("postgresql://localhost:5432/" + DB).Session()

    docs_path = "tests/data/html/"
    pdf_path = "tests/data/pdf/"

    # Parsing
    logger.info("Parsing...")
    doc_preprocessor = HTMLDocPreprocessor(docs_path, max_docs=max_docs)
    corpus_parser = Parser(
        session, structural=True, lingual=True, visual=True, pdf_path=pdf_path
    )
    corpus_parser.apply(doc_preprocessor, parallelism=PARALLEL)
    assert session.query(Document).count() == max_docs
    assert session.query(Sentence).count() == 799
    docs = session.query(Document).order_by(Document.name).all()

    # Mention Extraction
    part_ngrams = MentionNgramsPart(parts_by_doc=None, n_max=3)
    temp_ngrams = MentionNgramsTemp(n_max=2)

    Part = mention_subclass("Part")
    Temp = mention_subclass("Temp")

    mention_extractor = MentionExtractor(
        session, [Part, Temp], [part_ngrams, temp_ngrams], [part_matcher, temp_matcher]
    )
    mention_extractor.apply(docs, parallelism=PARALLEL)

    assert session.query(Part).count() == 70
    assert session.query(Temp).count() == 23
    part = session.query(Part).order_by(Part.id).all()[0]
    temp = session.query(Temp).order_by(Temp.id).all()[0]
    logger.info(f"Part: {part.context}")
    logger.info(f"Temp: {temp.context}")

    # Candidate Extraction
    PartTemp = candidate_subclass("PartTemp", [Part, Temp])

    # Test that no throttler in candidate extractor
    candidate_extractor = CandidateExtractor(session, [PartTemp])  # Pass, no throttler

    candidate_extractor.apply(docs, split=0, parallelism=PARALLEL)

    assert session.query(PartTemp).count() == 1610

    # Featurization based on default feature library
    featurizer = Featurizer(session, [PartTemp])

    # Test that featurization default feature library
    featurizer.apply(split=0, train=True, parallelism=PARALLEL)
    assert session.query(Feature).count() == 1610
    assert session.query(FeatureKey).count() == 2272

    # Example feature extractor
    def feat_ext(candidates):
        candidates = candidates if isinstance(candidates, list) else [candidates]
        for candidate in candidates:
            yield candidate.id, f"cand_id_{candidate.id}", 1

    # Featurization with one extra feature extractor
    feature_extractors = FeatureExtractor(customize_feature_funcs=[feat_ext])
    featurizer = Featurizer(session, [PartTemp], feature_extractors=feature_extractors)

    # Test that featurization default feature library with one extra feature extractor
    featurizer.apply(split=0, train=True, parallelism=PARALLEL)
    assert session.query(Feature).count() == 1610
    assert session.query(FeatureKey).count() == 3882

    # Featurization with only textual feature
    feature_extractors = FeatureExtractor(features=["textual"])
    featurizer = Featurizer(session, [PartTemp], feature_extractors=feature_extractors)

    # Test that featurization default feature library
    featurizer.apply(split=0, train=True, parallelism=PARALLEL)
    assert session.query(Feature).count() == 1610
    assert session.query(FeatureKey).count() == 706

    # Featurization with only tabular feature
    feature_extractors = FeatureExtractor(features=["tabular"])
    featurizer = Featurizer(session, [PartTemp], feature_extractors=feature_extractors)

    # Test that featurization default feature library
    featurizer.apply(split=0, train=True, parallelism=PARALLEL)
    assert session.query(Feature).count() == 1610
    assert session.query(FeatureKey).count() == 1360

    # Featurization with only structural feature
    feature_extractors = FeatureExtractor(features=["structural"])
    featurizer = Featurizer(session, [PartTemp], feature_extractors=feature_extractors)

    # Test that featurization default feature library
    featurizer.apply(split=0, train=True, parallelism=PARALLEL)
    assert session.query(Feature).count() == 1610
    assert session.query(FeatureKey).count() == 116

    # Featurization with only visual feature
    feature_extractors = FeatureExtractor(features=["visual"])
    featurizer = Featurizer(session, [PartTemp], feature_extractors=feature_extractors)

    # Test that featurization default feature library
    featurizer.apply(split=0, train=True, parallelism=PARALLEL)
    assert session.query(Feature).count() == 1610
    assert session.query(FeatureKey).count() == 90
