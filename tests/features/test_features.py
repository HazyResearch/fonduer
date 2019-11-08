import logging

from fonduer import Meta
from fonduer.candidates import CandidateExtractor, MentionExtractor, MentionNgrams
from fonduer.candidates.models import candidate_subclass, mention_subclass
from fonduer.features import FeatureExtractor, Featurizer
from fonduer.features.models import FeatureKey
from fonduer.parser import Parser
from fonduer.parser.models import Document, Sentence
from fonduer.parser.preprocessors import HTMLDocPreprocessor
from tests.shared.hardware_matchers import part_matcher, temp_matcher

logger = logging.getLogger(__name__)
DB = "feature_test"
# Use 127.0.0.1 instead of localhost (#351)
CONN_STRING = f"postgresql://127.0.0.1:5432/{DB}"


def test_feature_extraction():
    """Test extracting candidates from mentions from documents."""
    PARALLEL = 1

    max_docs = 1
    session = Meta.init(CONN_STRING).Session()

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
    part_ngrams = MentionNgrams(n_max=1)
    temp_ngrams = MentionNgrams(n_max=1)

    Part = mention_subclass("Part")
    Temp = mention_subclass("Temp")

    mention_extractor = MentionExtractor(
        session, [Part, Temp], [part_ngrams, temp_ngrams], [part_matcher, temp_matcher]
    )
    mention_extractor.apply(docs, parallelism=PARALLEL)

    assert docs[0].name == "112823"
    assert session.query(Part).count() == 58
    assert session.query(Temp).count() == 16
    part = session.query(Part).order_by(Part.id).all()[0]
    temp = session.query(Temp).order_by(Temp.id).all()[0]
    logger.info(f"Part: {part.context}")
    logger.info(f"Temp: {temp.context}")

    # Candidate Extraction
    PartTemp = candidate_subclass("PartTemp", [Part, Temp])

    candidate_extractor = CandidateExtractor(session, [PartTemp])

    candidate_extractor.apply(docs, split=0, parallelism=PARALLEL)

    n_cands = session.query(PartTemp).count()

    # Featurization based on default feature library
    featurizer = Featurizer(session, [PartTemp])

    # Test that featurization default feature library
    featurizer.apply(split=0, train=True, parallelism=PARALLEL)
    n_default_feats = session.query(FeatureKey).count()
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
    n_default_w_customized_features = session.query(FeatureKey).count()
    featurizer.clear(train=True)

    # Example spurious feature extractor
    def bad_feat_ext(candidates):
        raise RuntimeError()

    # Featurization with a spurious feature extractor
    feature_extractors = FeatureExtractor(customize_feature_funcs=[bad_feat_ext])
    featurizer = Featurizer(session, [PartTemp], feature_extractors=feature_extractors)

    # Test that featurization default feature library with one extra feature extractor
    logger.info("Featurizing with a spurious feature extractor...")
    featurizer.apply(split=0, train=True, parallelism=PARALLEL)
    featurizer.clear(train=True)

    # Featurization with only textual feature
    feature_extractors = FeatureExtractor(features=["textual"])
    featurizer = Featurizer(session, [PartTemp], feature_extractors=feature_extractors)

    # Test that featurization textual feature library
    featurizer.apply(split=0, train=True, parallelism=PARALLEL)
    n_textual_features = session.query(FeatureKey).count()
    featurizer.clear(train=True)

    # Featurization with only tabular feature
    feature_extractors = FeatureExtractor(features=["tabular"])
    featurizer = Featurizer(session, [PartTemp], feature_extractors=feature_extractors)

    # Test that featurization tabular feature library
    featurizer.apply(split=0, train=True, parallelism=PARALLEL)
    n_tabular_features = session.query(FeatureKey).count()
    featurizer.clear(train=True)

    # Featurization with only structural feature
    feature_extractors = FeatureExtractor(features=["structural"])
    featurizer = Featurizer(session, [PartTemp], feature_extractors=feature_extractors)

    # Test that featurization structural feature library
    featurizer.apply(split=0, train=True, parallelism=PARALLEL)
    n_structural_features = session.query(FeatureKey).count()
    featurizer.clear(train=True)

    # Featurization with only visual feature
    feature_extractors = FeatureExtractor(features=["visual"])
    featurizer = Featurizer(session, [PartTemp], feature_extractors=feature_extractors)

    # Test that featurization visual feature library
    featurizer.apply(split=0, train=True, parallelism=PARALLEL)
    n_visual_features = session.query(FeatureKey).count()
    featurizer.clear(train=True)

    assert (
        n_default_feats
        == n_textual_features
        + n_tabular_features
        + n_structural_features
        + n_visual_features
    )

    assert n_default_w_customized_features == n_default_feats + n_cands
