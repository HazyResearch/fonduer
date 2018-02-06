from fonduer.features.content_features import *
from fonduer.features.core_features import *
from fonduer.features.structural_features import *
from fonduer.features.table_features import *
from fonduer.features.visual_features import *


def get_all_feats(candidates):
    for id, f, v in get_core_feats(candidates):
        yield id, f, v
    for id, f, v in get_content_feats(candidates):
        yield id, f, v
    for id, f, v in get_structural_feats(candidates):
        yield id, f, v
    for id, f, v in get_table_feats(candidates):
        yield id, f, v
    for id, f, v in get_visual_feats(candidates):
        yield id, f, v
