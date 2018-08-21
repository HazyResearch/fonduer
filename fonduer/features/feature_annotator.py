from fonduer.features.features import get_all_feats
from fonduer.utils.annotator import Annotator


class FeatureAnnotator(Annotator):
    def __init__(self, candidate_type, **kwargs):
        super(FeatureAnnotator, self).__init__(
            candidate_type, annotation_type="feature", f=get_all_feats, **kwargs
        )
