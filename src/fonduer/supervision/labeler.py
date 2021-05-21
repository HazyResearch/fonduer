"""Fonduer labeler."""
import logging
from collections import defaultdict
from typing import (
    Any,
    Callable,
    Collection,
    DefaultDict,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np
from sqlalchemy import Table
from sqlalchemy.orm import Session

from fonduer.candidates.models import Candidate
from fonduer.parser.models import Document
from fonduer.supervision.models import GoldLabelKey, Label, LabelKey
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
    unshift_label_matrix,
    upsert_keys,
)

logger = logging.getLogger(__name__)

# Snorkel changed the label convention: ABSTAIN is now represented by -1 (used to be 0).
# Accordingly, user-defined labels should now be 0-indexed (used to be 1-indexed).
# Details can be found at https://github.com/snorkel-team/snorkel/pull/1309
ABSTAIN = -1


class Labeler(UDFRunner):
    """An operator to add Label Annotations to Candidates.

    :param session: The database session to use.
    :param candidate_classes: A list of candidate_subclasses to label.
    :param parallelism: The number of processes to use in parallel. Default 1.
    """

    def __init__(
        self,
        session: Session,
        candidate_classes: List[Type[Candidate]],
        parallelism: int = 1,
    ):
        """Initialize the Labeler."""
        super().__init__(
            session,
            LabelerUDF,
            parallelism=parallelism,
            candidate_classes=candidate_classes,
        )
        self.candidate_classes = candidate_classes
        self.lfs: List[List[Callable]] = []

    def update(
        self,
        docs: Collection[Document] = None,
        split: int = 0,
        lfs: List[List[Callable]] = None,
        parallelism: int = None,
        progress_bar: bool = True,
        table: Table = Label,
    ) -> None:
        """Update the labels of the specified candidates based on the provided LFs.

        :param docs: If provided, apply the updated LFs to all the candidates
            in these documents.
        :param split: If docs is None, apply the updated LFs to the candidates
            in this particular split.
        :param lfs: A list of lists of labeling functions to update. Each list
            should correspond with the candidate_classes used to initialize the
            Labeler.
        :param parallelism: How many threads to use for extraction. This will
            override the parallelism value used to initialize the Labeler if
            it is provided.
        :param progress_bar: Whether or not to display a progress bar. The
            progress bar is measured per document.
        :param table: A (database) table labels are written to.
            Takes `Label` (by default) or `GoldLabel`.
        """
        if lfs is None:
            raise ValueError("Please provide a list of lists of labeling functions.")

        if len(lfs) != len(self.candidate_classes):
            raise ValueError("Please provide LFs for each candidate class.")

        self.table = table

        self.apply(
            docs=docs,
            split=split,
            lfs=lfs,
            train=True,
            clear=False,
            parallelism=parallelism,
            progress_bar=progress_bar,
            table=table,
        )

    def apply(  # type: ignore
        self,
        docs: Collection[Document] = None,
        split: int = 0,
        train: bool = False,
        lfs: List[List[Callable]] = None,
        clear: bool = True,
        parallelism: int = None,
        progress_bar: bool = True,
        table: Table = Label,
    ) -> None:
        """Apply the labels of the specified candidates based on the provided LFs.

        :param docs: If provided, apply the LFs to all the candidates in these
            documents.
        :param split: If docs is None, apply the LFs to the candidates in this
            particular split.
        :param train: Whether or not to update the global key set of labels and
            the labels of candidates.
        :param lfs: A list of lists of labeling functions to apply. Each list
            should correspond with the candidate_classes used to initialize the
            Labeler.
        :param clear: Whether or not to clear the labels table before applying
            these LFs.
        :param parallelism: How many threads to use for extraction. This will
            override the parallelism value used to initialize the Labeler if
            it is provided.
        :param progress_bar: Whether or not to display a progress bar. The
            progress bar is measured per document.
        :param table: A (database) table labels are written to.
            Takes `Label` (by default) or `GoldLabel`.

        :raises ValueError: If labeling functions are not provided for each
            candidate class.
        """
        if lfs is None:
            raise ValueError("Please provide a list of labeling functions.")

        if len(lfs) != len(self.candidate_classes):
            raise ValueError("Please provide LFs for each candidate class.")

        self.lfs = lfs
        self.table = table
        if docs:
            # Call apply on the specified docs for all splits
            # TODO: split is int
            split = ALL_SPLITS  # type: ignore
            super().apply(
                docs,
                split=split,
                train=train,
                lfs=self.lfs,
                clear=clear,
                parallelism=parallelism,
                progress_bar=progress_bar,
                table=table,
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
                lfs=self.lfs,
                clear=clear,
                parallelism=parallelism,
                progress_bar=progress_bar,
                table=table,
            )
            # Needed to sync the bulk operations
            self.session.commit()

    def get_keys(self) -> List[LabelKey]:
        """Return a list of keys for the Labels.

        :return: List of LabelKeys.
        """
        return list(get_sparse_matrix_keys(self.session, LabelKey))

    def upsert_keys(
        self,
        keys: Iterable[Union[str, Callable]],
        candidate_classes: Optional[
            Union[Type[Candidate], List[Type[Candidate]]]
        ] = None,
    ) -> None:
        """Upsert the specified keys from LabelKeys.

        :param keys: A list of labeling functions to upsert.
        :param candidate_classes: A list of the Candidates to upsert the key for.
            If None, upsert the keys for all candidate classes associated with
            this Labeler.
        """
        # Make sure keys is iterable
        keys = keys if isinstance(keys, (list, tuple)) else [keys]

        # Make sure candidate_classes is iterable
        if candidate_classes:
            candidate_classes = (
                candidate_classes
                if isinstance(candidate_classes, (list, tuple))
                else [candidate_classes]
            )

            # Ensure only candidate classes associated with the labeler are used.
            candidate_classes = [
                _.__tablename__
                for _ in candidate_classes
                if _ in self.candidate_classes
            ]

            if len(candidate_classes) == 0:
                logger.warning(
                    "You didn't specify valid candidate classes for this Labeler."
                )
                return
        # If unspecified, just use all candidate classes
        else:
            candidate_classes = [_.__tablename__ for _ in self.candidate_classes]

        # build dict for use by utils
        key_map = dict()
        for key in keys:
            # Assume key is an LF
            if hasattr(key, "__name__"):
                key_map[key.__name__] = set(candidate_classes)
            elif hasattr(key, "name"):
                key_map[key.name] = set(candidate_classes)
            else:
                key_map[key] = set(candidate_classes)

        upsert_keys(self.session, LabelKey, key_map)

    def drop_keys(
        self,
        keys: Iterable[Union[str, Callable]],
        candidate_classes: Optional[
            Union[Type[Candidate], List[Type[Candidate]]]
        ] = None,
    ) -> None:
        """Drop the specified keys from LabelKeys.

        :param keys: A list of labeling functions to delete.
        :param candidate_classes: A list of the Candidates to drop the key for.
            If None, drops the keys for all candidate classes associated with
            this Labeler.
        """
        # Make sure keys is iterable
        keys = keys if isinstance(keys, (list, tuple)) else [keys]

        # Make sure candidate_classes is iterable
        if candidate_classes:
            candidate_classes = (
                candidate_classes
                if isinstance(candidate_classes, (list, tuple))
                else [candidate_classes]
            )

            # Ensure only candidate classes associated with the labeler are used.
            candidate_classes = [
                _.__tablename__
                for _ in candidate_classes
                if _ in self.candidate_classes
            ]

            if len(candidate_classes) == 0:
                logger.warning(
                    "You didn't specify valid candidate classes for this Labeler."
                )
                return
        # If unspecified, just use all candidate classes
        else:
            candidate_classes = [_.__tablename__ for _ in self.candidate_classes]

        # build dict for use by utils
        key_map = dict()
        for key in keys:
            # Assume key is an LF
            if hasattr(key, "__name__"):
                key_map[key.__name__] = set(candidate_classes)
            elif hasattr(key, "name"):
                key_map[key.name] = set(candidate_classes)
            else:
                key_map[key] = set(candidate_classes)

        drop_keys(self.session, LabelKey, key_map)

    def _add(self, session: Session, records_list: List[List[Dict[str, Any]]]) -> None:
        for records in records_list:
            batch_upsert_records(session, self.table, records)

    def clear(  # type: ignore
        self,
        train: bool,
        split: int,
        lfs: Optional[List[List[Callable]]] = None,
        table: Table = Label,
        **kwargs: Any,
    ) -> None:
        """Delete Labels of each class from the database.

        :param train: Whether or not to clear the LabelKeys.
        :param split: Which split of candidates to clear labels from.
        :param lfs: This parameter is ignored.
        :param table: A (database) table labels are cleared from.
            Takes `Label` (by default) or `GoldLabel`.
        """
        # Clear Labels for the candidates in the split passed in.
        logger.info(f"Clearing Labels (split {split})")

        if split == ALL_SPLITS:
            sub_query = self.session.query(Candidate.id).subquery()
        else:
            sub_query = (
                self.session.query(Candidate.id)
                .filter(Candidate.split == split)
                .subquery()
            )
        query = self.session.query(table).filter(table.candidate_id.in_(sub_query))
        query.delete(synchronize_session="fetch")

        # Delete all old annotation keys
        if train:
            key_table = LabelKey if table == Label else GoldLabelKey
            logger.debug(
                f"Clearing all {key_table.__name__}s from {self.candidate_classes}..."
            )
            drop_all_keys(self.session, key_table, self.candidate_classes)

    def clear_all(self, table: Table = Label) -> None:
        """Delete all Labels.

        :param table: A (database) table labels are cleared from.
            Takes `Label` (by default) or `GoldLabel`.
        """
        key_table = LabelKey if table == Label else GoldLabelKey
        logger.info(f"Clearing ALL {table.__name__}s and {key_table.__name__}s.")
        self.session.query(table).delete(synchronize_session="fetch")
        self.session.query(key_table).delete(synchronize_session="fetch")

    def _after_apply(
        self, train: bool = False, table: Table = Label, **kwargs: Any
    ) -> None:
        # Insert all Label Keys
        if train:
            key_map: DefaultDict[str, set] = defaultdict(set)
            for label in self.session.query(table).all():
                cand = label.candidate
                for key in label.keys:
                    key_map[key].add(cand.__class__.__tablename__)
            key_table = LabelKey if table == Label else GoldLabelKey
            self.session.query(key_table).delete(synchronize_session="fetch")
            # TODO: upsert is too much. insert is fine as all keys are deleted.
            upsert_keys(self.session, key_table, key_map)

    def get_gold_labels(
        self, cand_lists: List[List[Candidate]], annotator: Optional[str] = None
    ) -> List[np.ndarray]:
        """Load dense matrix of GoldLabels for each candidate_class.

        :param cand_lists: The candidates to get gold labels for.
        :param annotator: A specific annotator key to get labels for. Default
            None.
        :raises ValueError: If get_gold_labels is called before gold labels are
            loaded, the result will contain ABSTAIN values. We raise a
            ValueError to help indicate this potential mistake to the user.
        :return: A list of MxN dense matrix where M are the candidates and N is the
            annotators. If annotator is provided, return a list of Mx1 matrix.
        """
        gold_labels = [
            unshift_label_matrix(m)
            for m in get_sparse_matrix(
                self.session, GoldLabelKey, cand_lists, key=annotator
            )
        ]

        for cand_labels in gold_labels:
            if ABSTAIN in cand_labels:
                raise ValueError(
                    "Gold labels contain ABSTAIN labels. "
                    "Did you load gold labels beforehand?"
                )

        return gold_labels

    def get_label_matrices(self, cand_lists: List[List[Candidate]]) -> List[np.ndarray]:
        """Load dense matrix of Labels for each candidate_class.

        :param cand_lists: The candidates to get labels for.
        :return: A list of MxN dense matrix where M are the candidates and N is the
            labeling functions.
        """
        return [
            unshift_label_matrix(m)
            for m in get_sparse_matrix(self.session, LabelKey, cand_lists)
        ]


class LabelerUDF(UDF):
    """UDF for performing candidate extraction."""

    def __init__(
        self,
        candidate_classes: Union[Type[Candidate], List[Type[Candidate]]],
        **kwargs: Any,
    ):
        """Initialize the LabelerUDF."""
        self.candidate_classes = (
            candidate_classes
            if isinstance(candidate_classes, (list, tuple))
            else [candidate_classes]
        )
        super().__init__(**kwargs)

    def _f_gen(self, c: Candidate) -> Iterator[Tuple[int, str, int]]:
        """Convert lfs into a generator of id, name, and labels.

        In particular, catch verbose values and convert to integer ones.
        """
        lf_idx = self.candidate_classes.index(c.__class__)
        labels = lambda c: [
            (
                c.id,
                lf.__name__ if hasattr(lf, "__name__") else lf.name,  # type: ignore
                lf(c),
            )
            for lf in self.lfs[lf_idx]
        ]
        for cid, lf_key, label in labels(c):
            # Note: We assume if the LF output is an int, it is already
            # mapped correctly
            if isinstance(label, int):
                yield cid, lf_key, label + 1  # convert to {0, 1, ..., k}
            # None is a protected LF output value corresponding to ABSTAIN,
            # representing LF abstaining
            elif label is None:
                yield cid, lf_key, ABSTAIN + 1  # convert to {0, 1, ..., k}
            elif label in c.values:
                #  convert to {0, 1, ..., k}
                yield cid, lf_key, c.values.index(label) + 1
            else:
                raise ValueError(
                    f"Can't parse label value {label} for candidate values {c.values}"
                )

    def apply(  # type: ignore
        self,
        doc: Document,
        lfs: List[List[Callable]],
        table: Table = Label,
        **kwargs: Any,
    ) -> List[List[Dict[str, Any]]]:
        """Extract candidates from the given Context.

        :param doc: A document to process.
        :param lfs: The list of functions to use to generate labels.
        """
        logger.debug(f"Document: {doc}")

        if lfs is None:
            raise ValueError("Must provide lfs kwarg.")

        self.lfs = lfs

        # Get all the candidates in this doc that will be labeled
        cands_list = [
            getattr(doc, candidate_class.__tablename__ + "s")
            for candidate_class in self.candidate_classes
        ]

        records_list = [
            list(get_mapping(table, cands, self._f_gen)) for cands in cands_list
        ]
        return records_list
