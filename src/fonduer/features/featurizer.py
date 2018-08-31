import logging

from fonduer.candidates.models import Candidate
from fonduer.features.features import get_all_feats
from fonduer.features.models import Feature, FeatureKey
from fonduer.utils.udf import UDF, UDFRunner
from fonduer.utils.utils_udf import (
    ALL_SPLITS,
    add_keys,
    get_cands_list_from_split,
    get_docs_from_split,
    get_mapping,
    get_sparse_matrix,
)

logger = logging.getLogger(__name__)


class Featurizer(UDFRunner):
    """An operator to add Feature Annotations to Candidates."""

    def __init__(self, session, candidate_classes):
        """Initialize the Featurizer."""
        super(Featurizer, self).__init__(
            session, FeaturizerUDF, candidate_classes=candidate_classes
        )
        self.candidate_classes = candidate_classes

    def apply(self, docs=None, split=0, train=False, **kwargs):
        """Call the FeaturizerUDF."""
        if docs:
            # Call apply on the specified docs for all splits
            split = ALL_SPLITS
            super(Featurizer, self).apply(
                docs, split=split, train=train, bulk=True, **kwargs
            )
            # Needed to sync the bulk operations
            self.session.commit()
        else:
            # Only grab the docs containing candidates from the given split.
            split_docs = get_docs_from_split(
                self.session, self.candidate_classes, split
            )
            super(Featurizer, self).apply(
                split_docs, split=split, train=train, bulk=True, **kwargs
            )
            # Needed to sync the bulk operations
            self.session.commit()

    def drop_keys(self, keys):
        """Drop the specified keys from FeatureKeys."""
        # Make sure keys is iterable
        keys = keys if isinstance(keys, (list, tuple)) else [keys]

        # Remove the specified keys
        for key in keys:
            self.session.query(FeatureKey).filter(FeatureKey.name == key).delete()

    def clear(self, train=False, split=0, **kwargs):
        """Delete Features of each class from the database."""
        # Clear Features for the candidates in the split passed in.
        logger.info("Clearing Features (split {})".format(split))

        sub_query = (
            self.session.query(Candidate.id).filter(Candidate.split == split).subquery()
        )
        query = self.session.query(Feature).filter(Feature.candidate_id.in_(sub_query))
        query.delete(synchronize_session="fetch")

        # Delete all old annotation keys
        if train:
            logger.debug("Clearing all FeatureKey...")
            query = self.session.query(FeatureKey)
            query.delete(synchronize_session="fetch")

    def clear_all(self, **kwargs):
        """Delete all Features."""
        logger.info("Clearing ALL Features and FeatureKeys.")
        self.session.query(Feature).delete()
        self.session.query(FeatureKey).delete()

    def get_feature_matrices(self, cand_lists):
        """Load sparse matrix of Features for each candidate_class."""
        return get_sparse_matrix(self.session, FeatureKey, cand_lists)


class FeaturizerUDF(UDF):
    """UDF for performing candidate extraction."""

    def __init__(self, candidate_classes, **kwargs):
        """Initialize the FeaturizerUDF."""
        self.candidate_classes = (
            candidate_classes
            if isinstance(candidate_classes, (list, tuple))
            else [candidate_classes]
        )
        super(FeaturizerUDF, self).__init__(**kwargs)

    def get_table(self, **kwargs):
        return Feature

    def apply(self, doc, split, train, **kwargs):
        """Extract candidates from the given Context.

        :param doc: A document to process.
        :param split: Which split to use.
        :param train: Whether or not to insert new FeatureKeys.
        """
        logger.debug("Document: {}".format(doc))

        # Get all the candidates in this doc that will be featurized
        cands_list = get_cands_list_from_split(
            self.session, self.candidate_classes, doc, split
        )

        feature_keys = set()
        for cands in cands_list:
            yield from get_mapping(cands, get_all_feats, feature_keys)

        # Insert all Feature Keys
        if train:
            add_keys(self.session, FeatureKey, feature_keys)
