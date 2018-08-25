import logging

from scipy.sparse import csr_matrix

from fonduer.candidates.models import Candidate
from fonduer.features.features import get_all_feats
from fonduer.features.models import Feature, FeatureKey
from fonduer.meta import Meta
from fonduer.utils.udf import UDF, UDFRunner

logger = logging.getLogger(__name__)

_ALL_SPLITS = -1


class FAnnotator(UDFRunner):
    """An operator to add Feature Annotations to Candidates."""

    def __init__(self, session, candidate_classes):
        """Initialize the FAnnotator."""
        super(FAnnotator, self).__init__(
            FAnnotatorUDF, candidate_classes=candidate_classes
        )
        self.candidate_classes = candidate_classes
        self.session = session

    def apply(self, docs=None, split=0, train=False, **kwargs):
        """Call the FAnnotatorUDF."""
        if docs:
            # Call apply on the specified docs for all splits
            split = _ALL_SPLITS
            super(FAnnotator, self).apply(
                docs, split=split, train=train, bulk=True, **kwargs
            )
            # Needed to sync the bulk operations
            self.session.commit()
        else:
            # Only grab the docs containing candidates from the given split.
            sub_query = (
                self.session.query(Candidate.id)
                .filter(Candidate.split == split)
                .subquery()
            )
            split_docs = set()
            for candidate_class in self.candidate_classes:
                split_docs.update(
                    cand.document
                    for cand in self.session.query(candidate_class)
                    .filter(candidate_class.id.in_(sub_query))
                    .all()
                )
            super(FAnnotator, self).apply(
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

    def get_candidates(self, docs=None, split=0):
        """Return a generator of lists of candidates for the FAnnotator."""
        result = []
        if docs:
            # Get cands from all splits
            for candidate_class in self.candidate_classes:
                cands = (
                    self.session.query(candidate_class)
                    .filter(candidate_class.document_id in [doc.id for doc in docs])
                    .order_by(candidate_class.id)
                    .all()
                )
                result.append(cands)
        else:
            for candidate_class in self.candidate_classes:
                sub_query = (
                    self.session.query(Candidate.id)
                    .filter(Candidate.split == split)
                    .subquery()
                )
                cands = (
                    self.session.query(candidate_class)
                    .filter(candidate_class.id.in_(sub_query))
                    .order_by(candidate_class.id)
                    .all()
                )
                result.append(cands)
        return result

    def clear(self, session, train=False, split=0, **kwargs):
        """Delete Features of each class from the database."""
        # Clear Features for the candidates in the split passed in.
        logger.info("Clearing Features (split {})".format(split))

        sub_query = (
            session.query(Candidate.id).filter(Candidate.split == split).subquery()
        )
        query = session.query(Feature).filter(Feature.candidate_id.in_(sub_query))
        query.delete(synchronize_session="fetch")

        # Delete all old annotation keys
        if train:
            logger.debug("Clearing all FeatureKey...")
            query = session.query(FeatureKey)
            query.delete(synchronize_session="fetch")

    def get_table(self, **kwargs):
        return Feature

    def clear_all(self, **kwargs):
        """Delete all Features."""
        logger.info("Clearing ALL Features and FeatureKeys.")
        self.session.query(Feature).delete()
        self.session.query(FeatureKey).delete()

    def get_feature_matrices(self, cand_lists):
        """Load sparse matrix of Features for each candidate_class."""
        result = []
        cand_lists = (
            cand_lists if isinstance(cand_lists, (list, tuple)) else [cand_lists]
        )
        for cand_list in cand_lists:
            keys = [
                key.name
                for key in self.session.query(FeatureKey)
                .order_by(FeatureKey.name)
                .all()
            ]

            indptr = [0]
            indices = []
            data = []
            for cand in cand_list:
                if cand.features:
                    cand_keys = cand.features[0].keys
                    cand_values = cand.features[0].values
                    indices.extend(
                        [keys.index(key) for key in cand_keys if key in keys]
                    )
                    data.extend(
                        [
                            cand_values[i[0]]
                            for i in enumerate(cand_keys)
                            if i[1] in keys
                        ]
                    )

                indptr.append(len(indices))
            result.append(
                csr_matrix((data, indices, indptr), shape=(len(cand_list), len(keys)))
            )

        return result


class FAnnotatorUDF(UDF):
    """UDF for performing candidate extraction."""

    def __init__(self, candidate_classes, **kwargs):
        """Initialize the FAnnotatorUDF."""
        self.candidate_classes = (
            candidate_classes
            if isinstance(candidate_classes, (list, tuple))
            else [candidate_classes]
        )
        super(FAnnotatorUDF, self).__init__(**kwargs)

    def _add_FeatureKeys(self, keys):
        """Construct a FeatureKey from the key."""
        # Do nothing if empty
        if not keys:
            return

        # NOTE: If you pprint these values, it may look funny because of the
        # newlines and tabs as whitespace characters in these names. Use normal
        # print.
        existing_keys = [k.name for k in self.session.query(FeatureKey).all()]
        new_keys = [k for k in keys if k not in existing_keys]

        # Bulk insert all new feature keys
        if new_keys:
            Meta.engine.execute(
                FeatureKey.__table__.insert(), [{"name": key} for key in new_keys]
            )

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
        cands = []
        if split == _ALL_SPLITS:
            # Get cands from all splits
            for candidate_class in self.candidate_classes:
                cands.extend(
                    self.session.query(candidate_class)
                    .filter(candidate_class.document_id == doc.id)
                    .all()
                )
        else:
            # Get cands from the specified split
            for candidate_class in self.candidate_classes:
                cands.extend(
                    self.session.query(candidate_class)
                    .filter(candidate_class.document_id == doc.id)
                    .filter(candidate_class.split == split)
                    .all()
                )

        feature_keys = set()
        for cand in cands:
            feature_args = {"candidate_id": cand.id}
            keys = []
            values = []
            for cid, key_name, feature in get_all_feats(cand):
                if feature == 0:
                    continue
                keys.append(key_name)
                values.append(feature)

            # Assemble feature arguments
            feature_args["keys"] = keys
            feature_args["values"] = values

            feature_keys.update(keys)
            yield feature_args

        # Insert all Feature Keys
        if train:
            self._add_FeatureKeys(feature_keys)
