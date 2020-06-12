"""Fonduer feature extractor."""
from typing import Callable, Dict, Iterator, List, Tuple, Union

from fonduer.candidates.models import Candidate
from fonduer.features.feature_libs.structural_features import (
    extract_structural_features,
)
from fonduer.features.feature_libs.tabular_features import extract_tabular_features
from fonduer.features.feature_libs.textual_features import extract_textual_features
from fonduer.features.feature_libs.visual_features import extract_visual_features

FEATURES: Dict[str, Callable[[List[Candidate]], Iterator[Tuple[int, str, int]]]] = {
    "textual": extract_textual_features,
    "structural": extract_structural_features,
    "tabular": extract_tabular_features,
    "visual": extract_visual_features,
}


# Type alias for feature_func
Feature_func = Callable[[List[Candidate]], Iterator[Tuple[int, str, int]]]


class FeatureExtractor(object):
    """A class to extract features from candidates.

    :param features: a list of which Fonduer feature types to extract, defaults
        to ["textual", "structural", "tabular", "visual"]
    :param customize_feature_funcs: a list of customized feature extractors where the
        extractor takes a list of candidates as input and yield tuples
        of (candidate_id, feature, value), defaults to []
    """

    def __init__(
        self,
        features: List[str] = ["textual", "structural", "tabular", "visual"],
        customize_feature_funcs: Union[Feature_func, List[Feature_func]] = [],
    ) -> None:
        """Initialize FeatureExtractor."""
        if not isinstance(customize_feature_funcs, list):
            customize_feature_funcs = [customize_feature_funcs]

        self.feature_extractors: List[
            Callable[[List[Candidate]], Iterator[Tuple[int, str, int]]]
        ] = []
        for feature in features:
            if feature not in FEATURES:
                raise ValueError(f"Unrecognized feature type: {feature}")
            self.feature_extractors.append(FEATURES[feature])

        self.feature_extractors.extend(customize_feature_funcs)

    def extract(
        self, candidates: Union[List[Candidate], Candidate]
    ) -> Iterator[Tuple[int, str, int]]:
        """Extract features from candidates.

        :param candidates: A list of candidates to extract features from
        """
        candidates = candidates if isinstance(candidates, list) else [candidates]

        for feature_extractor in self.feature_extractors:
            for candidate_id, feature, value in feature_extractor(candidates):
                yield candidate_id, feature, value
