from fonduer.features.feature_libs.structural_features import (
    extract_structural_features,
)
from fonduer.features.feature_libs.tabular_features import extract_tabular_features
from fonduer.features.feature_libs.textual_features import extract_textual_features
from fonduer.features.feature_libs.visual_features import extract_visual_features

FEATURES = {
    "textual": extract_textual_features,
    "structural": extract_structural_features,
    "tabular": extract_tabular_features,
    "visual": extract_visual_features,
}


class FeatureExtractor(object):
    """A class to extract features from candidates.

    :param features: a list of which Fonduer feature types to extract, defaults
        to ["textual", "structural", "tabular", "visual"]
    :type features: list, optional
    :param customize_feature_funcs: a list of customized feature extractors where the
        extractor takes a list of candidates as input and yield tuples
        of (candidate_id, feature, value), defaults to []
    :type customize_feature_funcs: list, optional
    """

    def __init__(
        self,
        features=["textual", "structural", "tabular", "visual"],
        customize_feature_funcs=[],
    ):
        self.feature_extractors = []
        for feature in features:
            if feature not in FEATURES:
                raise ValueError(f"Unrecognized feature type: {feature}")
            self.feature_extractors.append(FEATURES[feature])

        self.feature_extractors.extend(customize_feature_funcs)

    def extract(self, candidates):
        """Extract features from candidates.

        :param candidates: A list of candidates to extract features from
        :type candidates: list
        """
        for feature_extractor in self.feature_extractors:
            for candidate_id, feature, value in feature_extractor(candidates):
                yield candidate_id, feature, value
