"""Fonduer featurizer."""
import itertools
import logging
from collections import defaultdict
from typing import (
    Any,
    Collection,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
    Union,
)

from scipy.sparse import csr_matrix
from sqlalchemy.orm import Session

from fonduer.candidates.models import Candidate
from fonduer.features.feature_extractors import FeatureExtractor
from fonduer.features.models import Feature, FeatureKey
from fonduer.parser.models.document import Document
from fonduer.utils.udf import UDF, UDFRunner
from fonduer.utils.utils_udf import (
    ALL_SPLITS,
    batch_upsert_records,
    drop_all_keys,
    drop_keys,
    get_docs_from_split,
    get_mapping,
    get_sparse_matrix,
    get_sparse_matrix_keys,
    upsert_keys,
)

logger = logging.getLogger(__name__)


class Featurizer(UDFRunner):
    """An operator to add Feature Annotations to Candidates.

    :param session: The database session to use.
    :param candidate_classes: A list of candidate_subclasses to featurize.
    :param parallelism: The number of processes to use in parallel. Default 1.
    """

    def __init__(
        self,
        session: Session,
        candidate_classes: List[Candidate],
        feature_extractors: FeatureExtractor = FeatureExtractor(),
        parallelism: int = 1,
    ) -> None:
        """Initialize the Featurizer."""
        super().__init__(
            session,
            FeaturizerUDF,
            parallelism=parallelism,
            candidate_classes=candidate_classes,
            feature_extractors=feature_extractors,
        )
        self.candidate_classes = candidate_classes

    def update(
        self,
        docs: Optional[Collection[Document]] = None,
        split: int = 0,
        parallelism: Optional[int] = None,
        progress_bar: bool = True,
    ) -> None:
        """Update the features of the specified candidates.

        :param docs: If provided, apply features to all the candidates in these
            documents.
        :param split: If docs is None, apply features to the candidates in this
            particular split.
        :param parallelism: How many threads to use for extraction. This will
            override the parallelism value used to initialize the Featurizer if
            it is provided.
        :param progress_bar: Whether or not to display a progress bar. The
            progress bar is measured per document.
        """
        self.apply(
            docs=docs,
            split=split,
            train=True,
            clear=False,
            parallelism=parallelism,
            progress_bar=progress_bar,
        )

    def apply(  # type: ignore
        self,
        docs: Optional[Collection[Document]] = None,
        split: int = 0,
        train: bool = False,
        clear: bool = True,
        parallelism: Optional[int] = None,
        progress_bar: bool = True,
    ) -> None:
        """Apply features to the specified candidates.

        :param docs: If provided, apply features to all the candidates in these
            documents.
        :param split: If docs is None, apply features to the candidates in this
            particular split.
        :param train: Whether or not to update the global key set of features
            and the features of candidates.
        :param clear: Whether or not to clear the features table before
            applying features.
        :param parallelism: How many threads to use for extraction. This will
            override the parallelism value used to initialize the Featurizer if
            it is provided.
        :param progress_bar: Whether or not to display a progress bar. The
            progress bar is measured per document.
        """
        if docs:
            # Call apply on the specified docs for all splits
            # TODO: split is int
            split = ALL_SPLITS  # type: ignore
            super().apply(
                docs,
                split=split,
                train=train,
                clear=clear,
                parallelism=parallelism,
                progress_bar=progress_bar,
            )
            # Needed to sync the bulk operations
            self.session.commit()
        else:
            # Only grab the docs containing candidates from the given split.
            split_docs = get_docs_from_split(
                self.session, self.candidate_classes, split
            )
            super().apply(
                split_docs,
                split=split,
                train=train,
                clear=clear,
                parallelism=parallelism,
                progress_bar=progress_bar,
            )
            # Needed to sync the bulk operations
            self.session.commit()

    def upsert_keys(
        self,
        keys: Iterable[str],
        candidate_classes: Union[Candidate, Iterable[Candidate], None] = None,
    ) -> None:
        """Upsert the specified keys to FeatureKey.

        :param keys: A list of FeatureKey names to upsert.
        :param candidate_classes: A list of the Candidates to upsert the key for.
            If None, upsert the keys for all candidate classes associated with
            this Featurizer.
        """
        # Make sure keys is iterable
        keys = keys if isinstance(keys, (list, tuple)) else [keys]

        # Make sure candidate_classes is iterable
        if candidate_classes:
            candidate_classes = (
                candidate_classes
                if isinstance(candidate_classes, Iterable)
                else [candidate_classes]
            )

            # Ensure only candidate classes associated with the featurizer
            # are used.
            candidate_classes = [
                _.__tablename__
                for _ in candidate_classes
                if _ in self.candidate_classes
            ]

            if len(candidate_classes) == 0:
                logger.warning(
                    "You didn't specify valid candidate classes for this featurizer."
                )
                return
        # If unspecified, just use all candidate classes
        else:
            candidate_classes = [_.__tablename__ for _ in self.candidate_classes]

        # build dict for use by utils
        key_map = dict()
        for key in keys:
            key_map[key] = set(candidate_classes)
        upsert_keys(self.session, FeatureKey, key_map)

    def drop_keys(
        self,
        keys: Iterable[str],
        candidate_classes: Union[Candidate, Iterable[Candidate], None] = None,
    ) -> None:
        """Drop the specified keys from FeatureKeys.

        :param keys: A list of FeatureKey names to delete.
        :param candidate_classes: A list of the Candidates to drop the key for.
            If None, drops the keys for all candidate classes associated with
            this Featurizer.
        """
        # Make sure keys is iterable
        keys = keys if isinstance(keys, (list, tuple)) else [keys]

        # Make sure candidate_classes is iterable
        if candidate_classes:
            candidate_classes = (
                candidate_classes
                if isinstance(candidate_classes, Iterable)
                else [candidate_classes]
            )

            # Ensure only candidate classes associated with the featurizer
            # are used.
            candidate_classes = [
                _.__tablename__
                for _ in candidate_classes
                if _ in self.candidate_classes
            ]

            if len(candidate_classes) == 0:
                logger.warning(
                    "You didn't specify valid candidate classes for this featurizer."
                )
                return
        # If unspecified, just use all candidate classes
        else:
            candidate_classes = [_.__tablename__ for _ in self.candidate_classes]

        # build dict for use by utils
        key_map = dict()
        for key in keys:
            key_map[key] = set(candidate_classes)

        drop_keys(self.session, FeatureKey, key_map)

    def get_keys(self) -> List[FeatureKey]:
        """Return a list of keys for the Features.

        :return: List of FeatureKeys.
        """
        return list(get_sparse_matrix_keys(self.session, FeatureKey))

    def _add(self, session: Session, records_list: List[List[Dict[str, Any]]]) -> None:
        # Make a flat list of all records from the list of list of records.
        # This helps reduce the number of queries needed to update.
        all_records = list(itertools.chain.from_iterable(records_list))
        batch_upsert_records(session, Feature, all_records)

    def clear(self, train: bool = False, split: int = 0) -> None:  # type: ignore
        """Delete Features of each class from the database.

        :param train: Whether or not to clear the FeatureKeys
        :param split: Which split of candidates to clear features from.
        """
        # Clear Features for the candidates in the split passed in.
        logger.info(f"Clearing Features (split {split})")

        if split == ALL_SPLITS:
            sub_query = self.session.query(Candidate.id).subquery()
        else:
            sub_query = (
                self.session.query(Candidate.id)
                .filter(Candidate.split == split)
                .subquery()
            )
        query = self.session.query(Feature).filter(Feature.candidate_id.in_(sub_query))
        query.delete(synchronize_session="fetch")

        # Delete all old annotation keys
        if train:
            logger.debug(f"Clearing all FeatureKeys from {self.candidate_classes}...")
            drop_all_keys(self.session, FeatureKey, self.candidate_classes)

    def clear_all(self) -> None:
        """Delete all Features."""
        logger.info("Clearing ALL Features and FeatureKeys.")
        self.session.query(Feature).delete(synchronize_session="fetch")
        self.session.query(FeatureKey).delete(synchronize_session="fetch")

    def _after_apply(self, train: bool = False, **kwargs: Any) -> None:
        # Insert all Feature Keys
        if train:
            key_map: DefaultDict[str, set] = defaultdict(set)
            for feature in self.session.query(Feature).all():
                cand = feature.candidate
                for key in feature.keys:
                    key_map[key].add(cand.__class__.__tablename__)
            self.session.query(FeatureKey).delete(synchronize_session="fetch")
            # TODO: upsert is too much. insert is fine as all keys are deleted.
            upsert_keys(self.session, FeatureKey, key_map)

    def get_feature_matrices(
        self, cand_lists: List[List[Candidate]]
    ) -> List[csr_matrix]:
        """Load sparse matrix of Features for each candidate_class.

        :param cand_lists: The candidates to get features for.
        :return: A list of MxN sparse matrix where M are the candidates and N is the
            features.
        """
        return get_sparse_matrix(self.session, FeatureKey, cand_lists)


class FeaturizerUDF(UDF):
    """UDF for performing candidate extraction."""

    def __init__(
        self,
        candidate_classes: Iterable[Type[Candidate]],
        feature_extractors: FeatureExtractor,
        **kwargs: Any,
    ) -> None:
        """Initialize the FeaturizerUDF."""
        self.candidate_classes = (
            candidate_classes
            if isinstance(candidate_classes, (list, tuple))
            else [candidate_classes]
        )

        self.feature_extractors = feature_extractors

        super().__init__(**kwargs)

    def apply(self, doc: Document, **kwargs: Any) -> List[List[Dict[str, Any]]]:
        """Extract candidates from the given Context.

        :param doc: A document to process.
        """
        logger.debug(f"Document: {doc}")

        # Get all the candidates in this doc that will be featurized
        cands_list = [
            getattr(doc, candidate_class.__tablename__ + "s")
            for candidate_class in self.candidate_classes
        ]

        records_list = [
            list(get_mapping(Feature, cands, self.feature_extractors.extract))
            for cands in cands_list
        ]
        return records_list
