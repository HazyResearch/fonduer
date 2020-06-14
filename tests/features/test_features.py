"""Fonduer featurization unit tests."""
import itertools
import logging

import pytest

from fonduer.candidates import MentionNgrams
from fonduer.candidates.candidates import CandidateExtractorUDF
from fonduer.candidates.mentions import MentionExtractorUDF
from fonduer.candidates.models import candidate_subclass, mention_subclass
from fonduer.features import FeatureExtractor
from fonduer.features.featurizer import FeaturizerUDF
from tests.candidates.test_candidates import parse_doc
from tests.shared.hardware_matchers import part_matcher, temp_matcher

logger = logging.getLogger(__name__)


def test_unary_relation_feature_extraction():
    """Test extracting unary candidates from mentions from documents."""
    docs_path = "tests/data/html/112823.html"
    pdf_path = "tests/data/pdf/112823.pdf"

    # Parsing
    doc = parse_doc(docs_path, "112823", pdf_path)
    assert len(doc.sentences) == 799

    # Mention Extraction
    part_ngrams = MentionNgrams(n_max=1)

    Part = mention_subclass("Part")

    mention_extractor_udf = MentionExtractorUDF([Part], [part_ngrams], [part_matcher])
    doc = mention_extractor_udf.apply(doc)

    assert doc.name == "112823"
    assert len(doc.parts) == 62
    part = doc.parts[0]
    logger.info(f"Part: {part.context}")

    # Candidate Extraction
    PartRel = candidate_subclass("PartRel", [Part])

    candidate_extractor_udf = CandidateExtractorUDF([PartRel], None, False, False, True)
    doc = candidate_extractor_udf.apply(doc, split=0)

    # Featurization based on default feature library
    featurizer_udf = FeaturizerUDF([PartRel], FeatureExtractor())

    # Test that featurization default feature library
    features_list = featurizer_udf.apply(doc)
    features = itertools.chain.from_iterable(features_list)
    key_set = set([key for feature in features for key in feature["keys"]])
    n_default_feats = len(key_set)

    # Featurization with only textual feature
    feature_extractors = FeatureExtractor(features=["textual"])
    featurizer_udf = FeaturizerUDF([PartRel], feature_extractors=feature_extractors)

    # Test that featurization textual feature library
    features_list = featurizer_udf.apply(doc)
    features = itertools.chain.from_iterable(features_list)
    key_set = set([key for feature in features for key in feature["keys"]])
    n_textual_features = len(key_set)

    # Featurization with only tabular feature
    feature_extractors = FeatureExtractor(features=["tabular"])
    featurizer_udf = FeaturizerUDF([PartRel], feature_extractors=feature_extractors)

    # Test that featurization tabular feature library
    features_list = featurizer_udf.apply(doc)
    features = itertools.chain.from_iterable(features_list)
    key_set = set([key for feature in features for key in feature["keys"]])
    n_tabular_features = len(key_set)

    # Featurization with only structural feature
    feature_extractors = FeatureExtractor(features=["structural"])
    featurizer_udf = FeaturizerUDF([PartRel], feature_extractors=feature_extractors)

    # Test that featurization structural feature library
    features_list = featurizer_udf.apply(doc)
    features = itertools.chain.from_iterable(features_list)
    key_set = set([key for feature in features for key in feature["keys"]])
    n_structural_features = len(key_set)

    # Featurization with only visual feature
    feature_extractors = FeatureExtractor(features=["visual"])
    featurizer_udf = FeaturizerUDF([PartRel], feature_extractors=feature_extractors)

    # Test that featurization visual feature library
    features_list = featurizer_udf.apply(doc)
    features = itertools.chain.from_iterable(features_list)
    key_set = set([key for feature in features for key in feature["keys"]])
    n_visual_features = len(key_set)

    assert (
        n_default_feats
        == n_textual_features
        + n_tabular_features
        + n_structural_features
        + n_visual_features
    )


def test_binary_relation_feature_extraction():
    """Test extracting candidates from mentions from documents."""
    docs_path = "tests/data/html/112823.html"
    pdf_path = "tests/data/pdf/112823.pdf"

    # Parsing
    doc = parse_doc(docs_path, "112823", pdf_path)
    assert len(doc.sentences) == 799

    # Mention Extraction
    part_ngrams = MentionNgrams(n_max=1)
    temp_ngrams = MentionNgrams(n_max=1)

    Part = mention_subclass("Part")
    Temp = mention_subclass("Temp")

    mention_extractor_udf = MentionExtractorUDF(
        [Part, Temp], [part_ngrams, temp_ngrams], [part_matcher, temp_matcher]
    )
    doc = mention_extractor_udf.apply(doc)

    assert len(doc.parts) == 62
    assert len(doc.temps) == 16
    part = doc.parts[0]
    temp = doc.temps[0]
    logger.info(f"Part: {part.context}")
    logger.info(f"Temp: {temp.context}")

    # Candidate Extraction
    PartTemp = candidate_subclass("PartTemp", [Part, Temp])

    candidate_extractor_udf = CandidateExtractorUDF(
        [PartTemp], None, False, False, True
    )

    doc = candidate_extractor_udf.apply(doc, split=0)

    # Manually set id as it is not set automatically b/c a database is not used.
    i = 0
    for cand in doc.part_temps:
        cand.id = i
        i = i + 1

    n_cands = len(doc.part_temps)

    # Featurization based on default feature library
    featurizer_udf = FeaturizerUDF([PartTemp], FeatureExtractor())

    # Test that featurization default feature library
    features_list = featurizer_udf.apply(doc)
    features = itertools.chain.from_iterable(features_list)
    key_set = set([key for feature in features for key in feature["keys"]])
    n_default_feats = len(key_set)

    # Example feature extractor
    def feat_ext(candidates):
        candidates = candidates if isinstance(candidates, list) else [candidates]
        for candidate in candidates:
            yield candidate.id, f"cand_id_{candidate.id}", 1

    # Featurization with one extra feature extractor
    feature_extractors = FeatureExtractor(customize_feature_funcs=[feat_ext])
    featurizer_udf = FeaturizerUDF([PartTemp], feature_extractors=feature_extractors)

    # Test that featurization default feature library with one extra feature extractor
    features_list = featurizer_udf.apply(doc)
    features = itertools.chain.from_iterable(features_list)
    key_set = set([key for feature in features for key in feature["keys"]])
    n_default_w_customized_features = len(key_set)

    # Example spurious feature extractor
    def bad_feat_ext(candidates):
        raise RuntimeError()

    # Featurization with a spurious feature extractor
    feature_extractors = FeatureExtractor(customize_feature_funcs=[bad_feat_ext])
    featurizer_udf = FeaturizerUDF([PartTemp], feature_extractors=feature_extractors)

    # Test that featurization default feature library with one extra feature extractor
    logger.info("Featurizing with a spurious feature extractor...")
    with pytest.raises(RuntimeError):
        features = featurizer_udf.apply(doc)

    # Featurization with only textual feature
    feature_extractors = FeatureExtractor(features=["textual"])
    featurizer_udf = FeaturizerUDF([PartTemp], feature_extractors=feature_extractors)

    # Test that featurization textual feature library
    features_list = featurizer_udf.apply(doc)
    features = itertools.chain.from_iterable(features_list)
    key_set = set([key for feature in features for key in feature["keys"]])
    n_textual_features = len(key_set)

    # Featurization with only tabular feature
    feature_extractors = FeatureExtractor(features=["tabular"])
    featurizer_udf = FeaturizerUDF([PartTemp], feature_extractors=feature_extractors)

    # Test that featurization tabular feature library
    features_list = featurizer_udf.apply(doc)
    features = itertools.chain.from_iterable(features_list)
    key_set = set([key for feature in features for key in feature["keys"]])
    n_tabular_features = len(key_set)

    # Featurization with only structural feature
    feature_extractors = FeatureExtractor(features=["structural"])
    featurizer_udf = FeaturizerUDF([PartTemp], feature_extractors=feature_extractors)

    # Test that featurization structural feature library
    features_list = featurizer_udf.apply(doc)
    features = itertools.chain.from_iterable(features_list)
    key_set = set([key for feature in features for key in feature["keys"]])
    n_structural_features = len(key_set)

    # Featurization with only visual feature
    feature_extractors = FeatureExtractor(features=["visual"])
    featurizer_udf = FeaturizerUDF([PartTemp], feature_extractors=feature_extractors)

    # Test that featurization visual feature library
    features_list = featurizer_udf.apply(doc)
    features = itertools.chain.from_iterable(features_list)
    key_set = set([key for feature in features for key in feature["keys"]])
    n_visual_features = len(key_set)

    assert (
        n_default_feats
        == n_textual_features
        + n_tabular_features
        + n_structural_features
        + n_visual_features
    )

    assert n_default_w_customized_features == n_default_feats + n_cands


def test_multary_relation_feature_extraction():
    """Test extracting candidates from mentions from documents."""
    docs_path = "tests/data/html/112823.html"
    pdf_path = "tests/data/pdf/112823.pdf"

    # Parsing
    doc = parse_doc(docs_path, "112823", pdf_path)
    assert len(doc.sentences) == 799

    # Mention Extraction
    part_ngrams = MentionNgrams(n_max=1)
    temp_ngrams = MentionNgrams(n_max=1)
    volt_ngrams = MentionNgrams(n_max=1)

    Part = mention_subclass("Part")
    Temp = mention_subclass("Temp")
    Volt = mention_subclass("Volt")

    mention_extractor_udf = MentionExtractorUDF(
        [Part, Temp, Volt],
        [part_ngrams, temp_ngrams, volt_ngrams],
        [part_matcher, temp_matcher, volt_matcher],
    )
    doc = mention_extractor_udf.apply(doc)

    assert len(doc.parts) == 62
    assert len(doc.temps) == 16
    assert len(doc.volts) == 33
    part = doc.parts[0]
    temp = doc.temps[0]
    volt = doc.volts[0]
    logger.info(f"Part: {part.context}")
    logger.info(f"Temp: {temp.context}")
    logger.info(f"Volt: {volt.context}")

    # Candidate Extraction
    PartTempVolt = candidate_subclass("PartTempVolt", [Part, Temp, Volt])

    candidate_extractor_udf = CandidateExtractorUDF(
        [PartTempVolt], None, False, False, True
    )

    doc = candidate_extractor_udf.apply(doc, split=0)

    # Manually set id as it is not set automatically b/c a database is not used.
    i = 0
    for cand in doc.part_temp_volts:
        cand.id = i
        i = i + 1

    n_cands = len(doc.part_temp_volts)

    # Featurization based on default feature library
    featurizer_udf = FeaturizerUDF([PartTempVolt], FeatureExtractor())

    # Test that featurization default feature library
    features_list = featurizer_udf.apply(doc)
    features = itertools.chain.from_iterable(features_list)
    key_set = set([key for feature in features for key in feature["keys"]])
    n_default_feats = len(key_set)

    # Example feature extractor
    def feat_ext(candidates):
        candidates = candidates if isinstance(candidates, list) else [candidates]
        for candidate in candidates:
            yield candidate.id, f"cand_id_{candidate.id}", 1

    # Featurization with one extra feature extractor
    feature_extractors = FeatureExtractor(
        features=["structural", "tabular", "visual"], customize_feature_funcs=[feat_ext]
    )
    featurizer_udf = FeaturizerUDF(
        [PartTempVolt], feature_extractors=feature_extractors
    )

    # Test that featurization default feature library with one extra feature extractor
    features_list = featurizer_udf.apply(doc)
    features = itertools.chain.from_iterable(features_list)
    key_set = set([key for feature in features for key in feature["keys"]])
    n_default_w_customized_features = len(key_set)

    # Example spurious feature extractor
    def bad_feat_ext(candidates):
        raise RuntimeError()

    # Featurization with a spurious feature extractor
    feature_extractors = FeatureExtractor(
        features=["structural", "tabular", "visual"],
        customize_feature_funcs=[bad_feat_ext],
    )
    featurizer_udf = FeaturizerUDF(
        [PartTempVolt], feature_extractors=feature_extractors
    )

    # Test that featurization default feature library with one extra feature extractor
    logger.info("Featurizing with a spurious feature extractor...")
    with pytest.raises(RuntimeError):
        features = featurizer_udf.apply(doc)

    # Featurization with only textual feature
    feature_extractors = FeatureExtractor(features=["textual"])
    featurizer_udf = FeaturizerUDF(
        [PartTempVolt], feature_extractors=feature_extractors
    )

    # Test that featurization textual feature library
    features_list = featurizer_udf.apply(doc)
    features = itertools.chain.from_iterable(features_list)
    key_set = set([key for feature in features for key in feature["keys"]])
    n_textual_features = len(key_set)

    # Featurization with only tabular feature
    feature_extractors = FeatureExtractor(features=["tabular"])
    featurizer_udf = FeaturizerUDF(
        [PartTempVolt], feature_extractors=feature_extractors
    )

    # Test that featurization tabular feature library
    features_list = featurizer_udf.apply(doc)
    features = itertools.chain.from_iterable(features_list)
    key_set = set([key for feature in features for key in feature["keys"]])
    n_tabular_features = len(key_set)

    # Featurization with only structural feature
    feature_extractors = FeatureExtractor(features=["structural"])
    featurizer_udf = FeaturizerUDF(
        [PartTempVolt], feature_extractors=feature_extractors
    )

    # Test that featurization structural feature library
    features_list = featurizer_udf.apply(doc)
    features = itertools.chain.from_iterable(features_list)
    key_set = set([key for feature in features for key in feature["keys"]])
    n_structural_features = len(key_set)

    # Featurization with only visual feature
    feature_extractors = FeatureExtractor(features=["visual"])
    featurizer_udf = FeaturizerUDF(
        [PartTempVolt], feature_extractors=feature_extractors
    )

    # Test that featurization visual feature library
    features_list = featurizer_udf.apply(doc)
    features = itertools.chain.from_iterable(features_list)
    key_set = set([key for feature in features for key in feature["keys"]])
    n_visual_features = len(key_set)

    assert (
        n_default_feats
        == n_textual_features
        + n_tabular_features
        + n_structural_features
        + n_visual_features
    )

    assert n_default_w_customized_features == n_default_feats + n_cands
