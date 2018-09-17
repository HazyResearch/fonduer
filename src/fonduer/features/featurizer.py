import logging

from fonduer.candidates.models import Candidate
from fonduer.features.feature_libs import get_all_feats
from fonduer.features.models import Feature, FeatureKey
from fonduer.utils.udf import UDF, UDFRunner
from fonduer.utils.utils_udf import (
    ALL_SPLITS,
    add_keys,
    batch_upsert_records,
    get_cands_list_from_split,
    get_docs_from_split,
    get_mapping,
    get_sparse_matrix,
    get_sparse_matrix_keys,
)

logger = logging.getLogger(__name__)


class Featurizer(UDFRunner):
    """An operator to add Feature Annotations to Candidates."""

    def __init__(self, session, candidate_classes, parallelism=1):
        """Initialize the Featurizer.

        :param session: The database session to use.
        :param candidate_classes: A list of candidate_subclasses to featurize.
        :param parallelism: The number of processes to use in parallel. Default 1.
        """
        super(Featurizer, self).__init__(
            session,
            FeaturizerUDF,
            parallelism=parallelism,
            candidate_classes=candidate_classes,
        )
        self.candidate_classes = candidate_classes

    def update(self, docs=None, split=0, parallelism=None, progress_bar=True):
        """Update the labels of the specified candidates based on the provided LFs.

        :param docs: If provided, apply features to all the candidates in these
            documents.
        :param split: If docs is None, apply features to the candidates in this
            particular split.
        :type split: int
        :param parallelism: How many threads to use for extraction. This will
            override the parallelism value used to initialize the Featurizer if
            it is provided.
        :type parallelism: int
        :param progress_bar: Whether or not to display a progress bar. The
            progress bar is measured per document.
        :type progress_bar: bool
        """
        self.apply(
            docs=docs,
            split=split,
            train=True,
            clear=False,
            parallelism=parallelism,
            progress_bar=progress_bar,
        )

    def apply(self, docs=None, split=0, train=False, clear=True, **kwargs):
        """Apply features to the specified candidates.

        :param docs: If provided, apply features to all the candidates in these
            documents.
        :param split: If docs is None, apply features to the candidates in this
            particular split.
        :param train: Whether or not to update the global key set of features and
            the features of candidates.
        :param clear: Whether or not to clear the features table before applying
            features.
        """
        if docs:
            # Call apply on the specified docs for all splits
            split = ALL_SPLITS
            super(Featurizer, self).apply(
                docs, split=split, train=train, clear=clear, **kwargs
            )
            # Needed to sync the bulk operations
            self.session.commit()
        else:
            # Only grab the docs containing candidates from the given split.
            split_docs = get_docs_from_split(
                self.session, self.candidate_classes, split
            )
            super(Featurizer, self).apply(
                split_docs, split=split, train=train, clear=clear, **kwargs
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

    def get_keys(self):
        """Return a list of keys for the Features."""
        return list(get_sparse_matrix_keys(self.session, FeatureKey))

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
            records = list(
                get_mapping(self.session, Feature, cands, get_all_feats, feature_keys)
            )
            batch_upsert_records(self.session, Feature, records)

        # Insert all Feature Keys
        if train:
            add_keys(self.session, FeatureKey, feature_keys)

        # This return + yield makes a completely empty generator
        return
        yield
