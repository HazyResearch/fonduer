import logging

from fonduer.candidates.models import Candidate
from fonduer.supervision.models import GoldLabelKey, Label, LabelKey
from fonduer.utils.udf import UDF, UDFRunner
from fonduer.utils.utils_udf import (
    ALL_SPLITS,
    batch_upsert_records,
    drop_all_keys,
    drop_keys,
    get_cands_list_from_split,
    get_docs_from_split,
    get_mapping,
    get_sparse_matrix,
    get_sparse_matrix_keys,
    upsert_keys,
)

logger = logging.getLogger(__name__)


def get_gold_labels(session, cand_lists, annotator_name="gold"):
    """Get the sparse matrix for the specified annotator.

    :param session: The database session.
    :param cand_lists: The candidates to get gold labels for.
    :type cand_lists: List of list of candidates.
    :param annotator: A specific annotator key to get labels for. Default
        "gold".
    :type annotator: str
    """
    return get_sparse_matrix(session, GoldLabelKey, cand_lists, key=annotator_name)


class Labeler(UDFRunner):
    """An operator to add Label Annotations to Candidates.

    :param session: The database session to use.
    :param candidate_classes: A list of candidate_subclasses to label.
    :type candidate_classes: list
    :param parallelism: The number of processes to use in parallel. Default 1.
    :type parallelism: int
    """

    def __init__(self, session, candidate_classes, parallelism=1):
        """Initialize the Labeler."""
        super(Labeler, self).__init__(
            session,
            LabelerUDF,
            parallelism=parallelism,
            candidate_classes=candidate_classes,
        )
        self.candidate_classes = candidate_classes
        self.lfs = []

    def update(self, docs=None, split=0, lfs=None, parallelism=None, progress_bar=True):
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
        :type parallelism: int
        :param progress_bar: Whether or not to display a progress bar. The
            progress bar is measured per document.
        :type progress_bar: bool
        """
        if lfs is None:
            raise ValueError("Please provide a list of lists of labeling functions.")

        if len(lfs) != len(self.candidate_classes):
            raise ValueError("Please provide LFs for each candidate class.")

        self.apply(
            docs=docs,
            split=split,
            lfs=lfs,
            train=True,
            clear=False,
            parallelism=parallelism,
            progress_bar=progress_bar,
        )

    def apply(
        self,
        docs=None,
        split=0,
        train=False,
        lfs=None,
        clear=True,
        parallelism=None,
        progress_bar=True,
    ):
        """Apply the labels of the specified candidates based on the provided LFs.

        :param docs: If provided, apply the LFs to all the candidates in these
            documents.
        :param split: If docs is None, apply the LFs to the candidates in this
            particular split.
        :type split: int
        :param train: Whether or not to update the global key set of labels and
            the labels of candidates.
        :type train: bool
        :param lfs: A list of lists of labeling functions to apply. Each list
            should correspond with the candidate_classes used to initialize the
            Labeler.
        :type lfs: list of lists
        :param clear: Whether or not to clear the labels table before applying
            these LFs.
        :type clear: bool
        :param parallelism: How many threads to use for extraction. This will
            override the parallelism value used to initialize the Labeler if
            it is provided.
        :type parallelism: int
        :param progress_bar: Whether or not to display a progress bar. The
            progress bar is measured per document.
        :type progress_bar: bool

        :raises ValueError: If labeling functions are not provided for each
            candidate class.
        """
        if lfs is None:
            raise ValueError("Please provide a list of labeling functions.")

        if len(lfs) != len(self.candidate_classes):
            raise ValueError("Please provide LFs for each candidate class.")

        self.lfs = lfs
        if docs:
            # Call apply on the specified docs for all splits
            split = ALL_SPLITS
            super(Labeler, self).apply(
                docs,
                split=split,
                train=train,
                lfs=self.lfs,
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
            super(Labeler, self).apply(
                split_docs,
                split=split,
                train=train,
                lfs=self.lfs,
                clear=clear,
                parallelism=parallelism,
                progress_bar=progress_bar,
            )
            # Needed to sync the bulk operations
            self.session.commit()

    def get_keys(self):
        """Return a list of keys for the Labels.

        :return: List of LabelKeys.
        :rtype: list
        """
        return list(get_sparse_matrix_keys(self.session, LabelKey))

    def drop_keys(self, keys, candidate_classes=None):
        """Drop the specified keys from LabelKeys.

        :param keys: A list of labeling functions to delete.
        :type keys: list, tuple
        :param candidate_classes: A list of the Candidates to drop the key for.
            If None, drops the keys for all candidate classes associated with
            this Labeler.
        :type candidate_classes: list, tuple
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
            try:
                key_map[key.__name__] = set(candidate_classes)
            except AttributeError:
                key_map[key] = set(candidate_classes)

        drop_keys(self.session, LabelKey, key_map)

    def clear(self, train, split, lfs=None):
        """Delete Labels of each class from the database.

        :param train: Whether or not to clear the LabelKeys.
        :type train: bool
        :param split: Which split of candidates to clear labels from.
        :type split: int
        :param lfs: This parameter is ignored.
        """
        # Clear Labels for the candidates in the split passed in.
        logger.info("Clearing Labels (split {})".format(split))

        sub_query = (
            self.session.query(Candidate.id).filter(Candidate.split == split).subquery()
        )
        query = self.session.query(Label).filter(Label.candidate_id.in_(sub_query))
        query.delete(synchronize_session="fetch")

        # Delete all old annotation keys
        if train:
            logger.debug(
                "Clearing all LabelKeys from {}...".format(self.candidate_classes)
            )
            drop_all_keys(self.session, LabelKey, self.candidate_classes)

    def clear_all(self):
        """Delete all Labels."""
        logger.info("Clearing ALL Labels and LabelKeys.")
        self.session.query(Label).delete()
        self.session.query(LabelKey).delete()

    def get_gold_labels(self, cand_lists, annotator=None):
        """Load sparse matrix of GoldLabels for each candidate_class.

        :param cand_lists: The candidates to get gold labels for.
        :type cand_lists: List of list of candidates.
        :param annotator: A specific annotator key to get labels for. Default
            None.
        :type annotator: str
        :return: An MxN sparse matrix where M are the candidates and N is the
            annotators. If annotator is provided, return an Mx1 matrix.
        :rtype: csr_matrix
        """
        return get_sparse_matrix(self.session, GoldLabelKey, cand_lists, key=annotator)

    def get_label_matrices(self, cand_lists):
        """Load sparse matrix of Labels for each candidate_class.

        :param cand_lists: The candidates to get labels for.
        :type cand_lists: List of list of candidates.
        :return: An MxN sparse matrix where M are the candidates and N is the
            labeling functions.
        :rtype: csr_matrix
        """
        return get_sparse_matrix(self.session, LabelKey, cand_lists)


class LabelerUDF(UDF):
    """UDF for performing candidate extraction."""

    def __init__(self, candidate_classes, **kwargs):
        """Initialize the LabelerUDF."""
        self.candidate_classes = (
            candidate_classes
            if isinstance(candidate_classes, (list, tuple))
            else [candidate_classes]
        )
        super(LabelerUDF, self).__init__(**kwargs)

    def _f_gen(self, c):
        """Convert lfs into a generator of id, name, and labels.

        In particular, catch verbose values and convert to integer ones.
        """
        lf_idx = self.candidate_classes.index(c.__class__)
        labels = lambda c: [(c.id, lf.__name__, lf(c)) for lf in self.lfs[lf_idx]]
        for cid, lf_key, label in labels(c):
            # Note: We assume if the LF output is an int, it is already
            # mapped correctly
            if isinstance(label, int):
                yield cid, lf_key, label
            # None is a protected LF output value corresponding to 0,
            # representing LF abstaining
            elif label is None:
                yield cid, lf_key, 0
            elif label in c.values:
                if c.cardinality > 2:
                    yield cid, lf_key, c.values.index(label) + 1
                # Note: Would be nice to not special-case here, but for
                # consistency we leave binary LF range as {-1,0,1}
                else:
                    val = 1 if c.values.index(label) == 0 else -1
                    yield cid, lf_key, val
            else:
                raise ValueError(
                    "Can't parse label value {} for candidate values {}".format(
                        label, c.values
                    )
                )

    def apply(self, doc, split, train, lfs, **kwargs):
        """Extract candidates from the given Context.

        :param doc: A document to process.
        :param split: Which split to use.
        :param train: Whether or not to insert new LabelKeys.
        :param lfs: The list of functions to use to generate labels.
        """
        logger.debug("Document: {}".format(doc))

        if lfs is None:
            raise ValueError("Must provide lfs kwarg.")

        self.lfs = lfs

        # Get all the candidates in this doc that will be labeled
        cands_list = get_cands_list_from_split(
            self.session, self.candidate_classes, doc, split
        )

        label_map = dict()
        for cands in cands_list:
            records = list(
                get_mapping(self.session, Label, cands, self._f_gen, label_map)
            )
            batch_upsert_records(self.session, Label, records)

        # Insert all Label Keys
        if train:
            upsert_keys(self.session, LabelKey, label_map)

        # This return + yield makes a completely empty generator
        return
        yield
