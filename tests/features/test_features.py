import logging

from fonduer import Meta
from fonduer.candidates import CandidateExtractor, MentionExtractor, MentionNgrams
from fonduer.candidates.models import candidate_subclass, mention_subclass
from fonduer.features import FeatureExtractor, Featurizer
from fonduer.features.models import Feature, FeatureKey
from fonduer.parser import Parser
from fonduer.parser.models import Document, Sentence
from fonduer.parser.preprocessors import HTMLDocPreprocessor
from tests.shared.hardware_matchers import part_matcher, temp_matcher
from tests.shared.hardware_throttlers import temp_throttler

logger = logging.getLogger(__name__)
DB = "feature_test"


def test_feature_extraction(caplog):
    """Test extracting candidates from mentions from documents."""
    caplog.set_level(logging.INFO)

    PARALLEL = 1

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
    part_ngrams = MentionNgrams(n_max=3)
    temp_ngrams = MentionNgrams(n_max=2)

    Part = mention_subclass("Part")
    Temp = mention_subclass("Temp")

    mention_extractor = MentionExtractor(
        session, [Part, Temp], [part_ngrams, temp_ngrams], [part_matcher, temp_matcher]
    )
    mention_extractor.apply(docs, parallelism=PARALLEL)

    assert session.query(Part).count() == 58
    assert session.query(Temp).count() == 16
    part = session.query(Part).order_by(Part.id).all()[0]
    temp = session.query(Temp).order_by(Temp.id).all()[0]
    logger.info(f"Part: {part.context}")
    logger.info(f"Temp: {temp.context}")

    # Candidate Extraction
    PartTemp = candidate_subclass("PartTemp", [Part, Temp])

    candidate_extractor = CandidateExtractor(
        session, [PartTemp], throttlers=[temp_throttler]
    )

    candidate_extractor.apply(docs, split=0, parallelism=PARALLEL)

    assert session.query(PartTemp).count() == 872

    # Featurization based on default feature library
    featurizer = Featurizer(session, [PartTemp])

    # Test that featurization default feature library
    featurizer.apply(split=0, train=True, parallelism=PARALLEL)
    assert session.query(Feature).count() == 872
    assert session.query(FeatureKey).count() == 1917
    featurizer.clear(train=True)

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
    assert session.query(Feature).count() == 872
    assert session.query(FeatureKey).count() == 2789
    featurizer.clear(train=True)

    # Featurization with only textual feature
    feature_extractors = FeatureExtractor(features=["textual"])
    featurizer = Featurizer(session, [PartTemp], feature_extractors=feature_extractors)

    # Test that featurization default feature library
    featurizer.apply(split=0, train=True, parallelism=PARALLEL)
    assert session.query(Feature).count() == 872
    assert session.query(FeatureKey).count() == 581
    featurizer.clear(train=True)

    # Featurization with only tabular feature
    feature_extractors = FeatureExtractor(features=["tabular"])
    featurizer = Featurizer(session, [PartTemp], feature_extractors=feature_extractors)

    # Test that featurization default feature library
    featurizer.apply(split=0, train=True, parallelism=PARALLEL)
    assert session.query(Feature).count() == 872
    assert session.query(FeatureKey).count() == 1184
    featurizer.clear(train=True)

    # Featurization with only structural feature
    feature_extractors = FeatureExtractor(features=["structural"])
    featurizer = Featurizer(session, [PartTemp], feature_extractors=feature_extractors)

    # Test that featurization default feature library
    featurizer.apply(split=0, train=True, parallelism=PARALLEL)
    assert session.query(Feature).count() == 872
    assert session.query(FeatureKey).count() == 92
    featurizer.clear(train=True)

    # Featurization with only visual feature
    feature_extractors = FeatureExtractor(features=["visual"])
    featurizer = Featurizer(session, [PartTemp], feature_extractors=feature_extractors)

    # Test that featurization default feature library
    featurizer.apply(split=0, train=True, parallelism=PARALLEL)
    assert session.query(Feature).count() == 872
    assert session.query(FeatureKey).count() == 60
    featurizer.clear(train=True)
