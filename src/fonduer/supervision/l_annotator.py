import logging

from scipy.sparse import csr_matrix
from sqlalchemy.sql.expression import bindparam

from fonduer.candidates.models import Candidate
from fonduer.meta import Meta
from fonduer.supervision.models import GoldLabelKey, Label, LabelKey
from fonduer.utils.udf import UDF, UDFRunner

logger = logging.getLogger(__name__)

_ALL_SPLITS = -1


class LAnnotator(UDFRunner):
    """An operator to add Label Annotations to Candidates."""

    def __init__(self, session, candidate_classes):
        """Initialize the LAnnotator."""
        super(LAnnotator, self).__init__(
            LAnnotatorUDF, candidate_classes=candidate_classes
        )
        self.candidate_classes = candidate_classes
        self.session = session

    def update(self, docs=None, split=0, lfs=None, **kwargs):
        """Call the LAnnotatorUDF."""
        if lfs is None:
            raise ValueError("Please provide a list of labeling functions.")

        # Only take the non-updated LFs
        self.lfs = [
            lf for lf in self.lfs if lf.__name__ not in [_.__name__ for _ in lfs]
        ]

        # Then add the updated/new LFs
        self.lfs.extend(lfs)

        if docs:
            # Call apply on the specified docs for all splits
            split = _ALL_SPLITS
            super(LAnnotator, self).apply(
                docs,
                split=split,
                clear=False,
                train=False,
                update=True,
                bulk=True,
                lfs=self.lfs,
                **kwargs
            )
            # Needed to sync the bulk update
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
            super(LAnnotator, self).apply(
                split_docs,
                split=split,
                clear=False,
                train=False,
                update=True,
                bulk=True,
                lfs=self.lfs,
                **kwargs
            )
            # Needed to sync the bulk update
            self.session.commit()

    def apply(self, docs=None, split=0, train=False, lfs=None, **kwargs):
        """Call the LAnnotatorUDF."""
        if lfs is None:
            raise ValueError("Please provide a list of labeling functions.")

        self.lfs = lfs
        if docs:
            # Call apply on the specified docs for all splits
            split = _ALL_SPLITS
            super(LAnnotator, self).apply(
                docs,
                split=split,
                train=train,
                bulk=True,
                update=False,
                lfs=self.lfs,
                **kwargs
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
            super(LAnnotator, self).apply(
                split_docs,
                split=split,
                train=train,
                bulk=True,
                update=False,
                lfs=self.lfs,
                **kwargs
            )
            # Needed to sync the bulk operations
            self.session.commit()

    def get_lfs(self):
        """Return a list of labeling functions for this LAnnotator."""
        return self.lfs

    def drop_keys(self, keys):
        """Drop the specified keys from LabelKeys."""
        # Make sure keys is iterable
        keys = keys if isinstance(keys, (list, tuple)) else [keys]

        # Remove the specified keys
        for key in keys:
            self.session.query(LabelKey).filter(LabelKey.name == key).delete()

    def get_candidates(self, docs=None, split=0):
        """Return a generator of lists of candidates for the LAnnotator."""
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

    def get_table(self, **kwargs):
        return Label

    def clear_all(self, **kwargs):
        """Delete all Labels."""
        logger.info("Clearing ALL Labels and LabelKeys.")
        self.session.query(Label).delete()
        self.session.query(LabelKey).delete()

    def get_gold_labels(self, cand_lists, annotator=None):
        """Load sparse matrix of GoldLabels for each candidate_class."""
        result = []
        cand_lists = (
            cand_lists if isinstance(cand_lists, (list, tuple)) else [cand_lists]
        )
        for cand_list in cand_lists:
            if annotator:
                keys = [annotator]
            else:
                keys = [
                    key.name
                    for key in self.session.query(GoldLabelKey)
                    .order_by(GoldLabelKey.name)
                    .all()
                ]

            indptr = [0]
            indices = []
            data = []
            for cand in cand_list:
                if cand.labels:
                    cand_keys = cand.gold_labels[0].keys
                    cand_values = cand.gold_labels[0].values
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

            result.append(csr_matrix((data, indices, indptr)))

        return result

    def get_label_matrices(self, cand_lists):
        """Load sparse matrix of Labels for each candidate_class."""
        result = []
        cand_lists = (
            cand_lists if isinstance(cand_lists, (list, tuple)) else [cand_lists]
        )
        for cand_list in cand_lists:
            keys = [
                key.name
                for key in self.session.query(LabelKey).order_by(LabelKey.name).all()
            ]

            indptr = [0]
            indices = []
            data = []
            for cand in cand_list:
                if cand.labels:
                    cand_keys = cand.labels[0].keys
                    cand_values = cand.labels[0].values
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


class LAnnotatorUDF(UDF):
    """UDF for performing candidate extraction."""

    def __init__(self, candidate_classes, **kwargs):
        """Initialize the LAnnotatorUDF."""
        self.candidate_classes = (
            candidate_classes
            if isinstance(candidate_classes, (list, tuple))
            else [candidate_classes]
        )
        super(LAnnotatorUDF, self).__init__(**kwargs)

    def _add_LabelKeys(self, keys):
        """Bulk insert the specified LabelKeys."""
        # Do nothing if empty
        if not keys:
            return

        existing_keys = [k.name for k in self.session.query(LabelKey).all()]
        new_keys = [k for k in keys if k not in existing_keys]

        # Bulk insert all new label keys
        if new_keys:
            Meta.engine.execute(
                LabelKey.__table__.insert(), [{"name": key} for key in new_keys]
            )

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

    # Convert lfs to a generator function
    # In particular, catch verbose values and convert to integer ones
    def _f_gen(self, labels, c):
        for lf_key, label in labels(c):
            # Note: We assume if the LF output is an int, it is already
            # mapped correctly
            if isinstance(label, int):
                yield lf_key, label
            # None is a protected LF output value corresponding to 0,
            # representing LF abstaining
            elif label is None:
                yield lf_key, 0
            elif label in c.values:
                if c.cardinality > 2:
                    yield lf_key, c.values.index(label) + 1
                # Note: Would be nice to not special-case here, but for
                # consistency we leave binary LF range as {-1,0,1}
                else:
                    val = 1 if c.values.index(label) == 0 else -1
                    yield lf_key, val
            else:
                raise ValueError(
                    "Can't parse label value {} for candidate values {}".format(
                        label, c.values
                    )
                )

    def _update(self, doc, split, lfs):
        """Perform an incremental add/update."""
        labels = lambda c: [(lf.__name__, lf(c)) for lf in lfs]

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

        label_keys = set()
        updates = []
        for cand in cands:
            label_args = {"candidate_id": cand.id}
            keys = []
            values = []
            for key_name, label in self._f_gen(labels, cand):
                if label == 0:
                    continue
                keys.append(key_name)
                values.append(label)

            # Assemble label arguments
            label_args["keys"] = keys
            label_args["values"] = values

            label_keys.update(keys)

            # If candidate exists, update keys, values
            if self.session.query(Label).filter(Label.candidate_id == cand.id).first():
                # Used as a WHERE argument for update
                label_args["_id"] = cand.id
                updates.append(label_args)
                continue

            # else, just insert
            yield label_args

        # Execute all updates
        self._update_labels(updates)

        # Insert all Label Keys
        self._add_LabelKeys(label_keys)

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

        if update:
            yield from self._update(doc, split, lfs)
        else:
            labels = lambda c: [(lf.__name__, lf(c)) for lf in lfs]

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

            label_keys = set()
            for cand in cands:
                label_args = {"candidate_id": cand.id}
                keys = []
                values = []
                for key_name, label in self._f_gen(labels, cand):
                    if label == 0:
                        continue
                    keys.append(key_name)
                    values.append(label)

                # Assemble label arguments
                label_args["keys"] = keys
                label_args["values"] = values

                label_keys.update(keys)
                yield label_args

            # Insert all Label Keys
            if train:
                self._add_LabelKeys(label_keys)
