from fonduer.features.feature_libs.content_features import get_content_feats
from fonduer.features.feature_libs.core_features import get_core_feats
from fonduer.features.feature_libs.structural_features import get_structural_feats
from fonduer.features.feature_libs.table_features import get_table_feats
from fonduer.features.feature_libs.visual_features import get_visual_feats


def get_all_feats(candidates):
    for candidate_id, feature, value in get_core_feats(candidates):
        yield candidate_id, feature, value
    for candidate_id, feature, value in get_content_feats(candidates):
        yield candidate_id, feature, value
    for candidate_id, feature, value in get_structural_feats(candidates):
        yield candidate_id, feature, value
    for candidate_id, feature, value in get_table_feats(candidates):
        yield candidate_id, feature, value
    for candidate_id, feature, value in get_visual_feats(candidates):
        yield candidate_id, feature, value


__all__ = [
    "get_all_feats",
    "get_content_feats",
    "get_core_feats",
    "get_structural_feats",
    "get_table_feats",
    "get_visual_feats",
]
