import logging

from sqlalchemy.sql.expression import bindparam

from fonduer.candidates.models import Candidate
from fonduer.meta import Meta
from fonduer.supervision.models import GoldLabelKey, Label, LabelKey
from fonduer.utils.udf import UDF, UDFRunner
from fonduer.utils.utils_udf import (
    ALL_SPLITS,
    add_keys,
    cands_from_split,
    docs_from_split,
    get_mapping,
    get_sparse_matrix,
)

logger = logging.getLogger(__name__)


class Labeler(UDFRunner):
    """An operator to add Label Annotations to Candidates."""

    def __init__(self, session, candidate_classes):
        """Initialize the Labeler."""
        super(Labeler, self).__init__(LabelerUDF, candidate_classes=candidate_classes)
        self.candidate_classes = candidate_classes
        self.session = session
        self.lfs = []

    def update(self, docs=None, split=0, lfs=None, **kwargs):
        """Call apply with update=True."""
        if lfs is None:
            raise ValueError("Please provide a list of labeling functions.")

        # Grab only the new/update LFS
        self.lfs = [
            lf for lf in self.lfs if lf.__name__ not in [_.__name__ for _ in lfs]
        ]
        # Then add the updated/new LFs
        self.lfs.extend(lfs)

        self.apply(
            docs=docs, split=split, lfs=self.lfs, train=False, update=True, **kwargs
        )

    def apply(self, docs=None, split=0, train=False, lfs=None, update=False, **kwargs):
        """Call the LabelerUDF."""
        if lfs is None:
            raise ValueError("Please provide a list of labeling functions.")

        self.lfs = lfs
        if docs:
            # Call apply on the specified docs for all splits
            split = ALL_SPLITS
            super(Labeler, self).apply(
                docs,
                split=split,
                train=train,
                bulk=True,
                update=update,
                lfs=self.lfs,
                **kwargs
            )
            # Needed to sync the bulk operations
            self.session.commit()
        else:
            # Only grab the docs containing candidates from the given split.
            split_docs = docs_from_split(self.session, self.candidate_classes, split)
            super(Labeler, self).apply(
                split_docs,
                split=split,
                train=train,
                bulk=True,
                update=update,
                lfs=self.lfs,
                **kwargs
            )
            # Needed to sync the bulk operations
            self.session.commit()

    def get_lfs(self):
        """Return a list of labeling functions for this Labeler."""
        return self.lfs

    def drop_keys(self, keys):
        """Drop the specified keys from LabelKeys."""
        # Make sure keys is iterable
        keys = keys if isinstance(keys, (list, tuple)) else [keys]

        # Remove the specified keys
        for key in keys:
            try:  # Assume key is an LF
                self.session.query(LabelKey).filter(
                    LabelKey.name == key.__name__
                ).delete()
            except AttributeError:
                self.session.query(LabelKey).filter(LabelKey.name == key).delete()

    def clear(self, session, train=False, split=0, **kwargs):
        """Delete Labels of each class from the database."""
        # Clear Labels for the candidates in the split passed in.
        logger.info("Clearing Labels (split {})".format(split))

        sub_query = (
            session.query(Candidate.id).filter(Candidate.split == split).subquery()
        )
        query = session.query(Label).filter(Label.candidate_id.in_(sub_query))
        query.delete(synchronize_session="fetch")

        # Delete all old annotation keys
        if train:
            logger.debug("Clearing all LabelKey...")
            query = session.query(LabelKey)
            query.delete(synchronize_session="fetch")

    def clear_all(self, **kwargs):
        """Delete all Labels."""
        logger.info("Clearing ALL Labels and LabelKeys.")
        self.session.query(Label).delete()
        self.session.query(LabelKey).delete()

    def get_gold_labels(self, cand_lists, annotator=None):
        """Load sparse matrix of GoldLabels for each candidate_class."""
        return get_sparse_matrix(self.session, GoldLabelKey, cand_lists, key=annotator)

    def get_label_matrices(self, cand_lists):
        """Load sparse matrix of Labels for each candidate_class."""
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

    def _update_labels(self, labels):
        """Bulk update the specified labels."""
        # Do nothing if empty
        if not labels:
            return

        Meta.engine.execute(
            Label.__table__.update()
            .where(Label.candidate_id == bindparam("_id"))
            .values({"keys": bindparam("keys"), "values": bindparam("values")}),
            [label for label in labels],
        )

    def get_table(self, **kwargs):
        return Label

    def _f_gen(self, c):
        """Convert lfs into a generator of id, name, and labels.

        In particular, catch verbose values and convert to integer ones.
        """
        labels = lambda c: [(c.id, lf.__name__, lf(c)) for lf in self.lfs]
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

    def apply(self, doc, split, train, update, lfs, **kwargs):
        """Extract candidates from the given Context.

        :param doc: A document to process.
        :param split: Which split to use.
        :param train: Whether or not to insert new LabelKeys.
        :param update: Whether to incrementally add/update new values.
        :param lfs: The list of functions to use to generate labels.
        """
        logger.debug("Document: {}".format(doc))

        if lfs is None:
            raise ValueError("Must provide lfs kwarg.")

        self.lfs = lfs

        # Get all the candidates in this doc that will be featurized
        cands = cands_from_split(self.session, self.candidate_classes, doc, split)

        label_keys = set()
        updates = []
        for label_args in get_mapping(cands, self._f_gen, label_keys):

            # If candidate exists, update keys, values
            if (
                self.session.query(Label)
                .filter(Label.candidate_id == label_args["candidate_id"])
                .first()
            ):
                # Used as a WHERE argument for update
                label_args["_id"] = label_args["candidate_id"]
                updates.append(label_args)
                continue

            # else, just insert
            yield label_args

        # Execute all updates
        self._update_labels(updates)

        # Insert all Label Keys
        if train or update:
            add_keys(self.session, LabelKey, label_keys)
