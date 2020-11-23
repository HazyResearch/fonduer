"""Fonduer candidate."""
import logging
from builtins import range
from itertools import product
from typing import (
    Any,
    Callable,
    Collection,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from sqlalchemy.orm import Session

from fonduer.candidates.models import Candidate, Mention
from fonduer.parser.models.document import Document
from fonduer.utils.udf import UDF, UDFRunner
from fonduer.utils.utils import get_set_of_stable_ids

logger = logging.getLogger(__name__)


class CandidateExtractor(UDFRunner):
    """An operator to extract Candidate objects from a Context.

    :Example:

        Assuming we have already defined a Part and Temp ``Mention`` subclass,
        and a throttler called templ_throttler, we can create a candidate
        extractor as follows::

            PartTemp = candidate_subclass("PartTemp", [Part, Temp])
            candidate_extractor = CandidateExtractor(
                session, [PartTemp], throttlers=[temp_throttler]
            )

    :param session: An initialized database session.
    :param candidate_classes: The types of relation to extract, defined using
        :func:`fonduer.candidates.candidate_subclass`.
    :param throttlers: optional functions for filtering out candidates
        which returns a Boolean expressing whether or not the candidate should
        be instantiated.
    :type throttlers: list of throttlers.
    :param self_relations: Boolean indicating whether to extract Candidates
        that relate the same context. Only applies to binary relations.
    :param nested_relations: Boolean indicating whether to extract Candidates
        that relate one Context with another that contains it. Only applies to
        binary relations.
    :param symmetric_relations: Boolean indicating whether to extract symmetric
        Candidates, i.e., rel(A,B) and rel(B,A), where A and B are Contexts.
        Only applies to binary relations.
    :param parallelism: The number of processes to use in parallel for calls
        to apply().
    :raises ValueError: If throttlers are provided, but a throtters are not the
        same length as candidate classes.
    """

    def __init__(
        self,
        session: Session,
        candidate_classes: List[Type[Candidate]],
        throttlers: Optional[List[Callable[[Tuple[Mention, ...]], bool]]] = None,
        self_relations: bool = False,
        nested_relations: bool = False,
        symmetric_relations: bool = True,
        parallelism: int = 1,
    ) -> None:
        """Set throttlers match candidate_classes if not provide."""
        if throttlers is None:
            throttlers = [None] * len(candidate_classes)

        """Initialize the CandidateExtractor."""
        super().__init__(
            session,
            CandidateExtractorUDF,
            parallelism=parallelism,
            candidate_classes=candidate_classes,
            throttlers=throttlers,
            self_relations=self_relations,
            nested_relations=nested_relations,
            symmetric_relations=symmetric_relations,
        )
        # Check that arity is sensible
        if len(candidate_classes) != len(throttlers):
            raise ValueError(
                "Provided different number of throttlers and candidate classes."
            )

        self.candidate_classes = candidate_classes

    def apply(  # type: ignore
        self,
        docs: Collection[Document],
        split: int = 0,
        clear: bool = True,
        parallelism: Optional[int] = None,
        progress_bar: bool = True,
    ) -> None:
        """Run the CandidateExtractor.

        :Example: To extract candidates from a set of training documents using
            4 cores::

                candidate_extractor.apply(train_docs, split=0, parallelism=4)

        :param docs: Set of documents to extract from.
        :param split: Which split to assign the extracted Candidates to.
        :param clear: Whether or not to clear the existing Candidates
            beforehand.
        :param parallelism: How many threads to use for extraction. This will
            override the parallelism value used to initialize the
            CandidateExtractor if it is provided.
        :param progress_bar: Whether or not to display a progress bar. The
            progress bar is measured per document.
        """
        super().apply(
            docs,
            split=split,
            clear=clear,
            parallelism=parallelism,
            progress_bar=progress_bar,
        )

    def clear(self, split: int) -> None:  # type: ignore
        """Clear Candidates of each class.

        Delete Candidates of each class initialized with the CandidateExtractor
        from the given split in the database.

        :param split: Which split to clear.
        """
        for candidate_class in self.candidate_classes:
            logger.info(
                f"Clearing table {candidate_class.__tablename__} (split {split})"
            )
            self.session.query(Candidate).filter(
                Candidate.type == candidate_class.__tablename__
            ).filter(Candidate.split == split).delete(synchronize_session="fetch")

    def clear_all(self, split: int) -> None:
        """Delete ALL Candidates from given split the database.

        :param split: Which split to clear.
        """
        logger.info("Clearing ALL Candidates.")
        self.session.query(Candidate).filter(Candidate.split == split).delete(
            synchronize_session="fetch"
        )

    def get_candidates(
        self,
        docs: Union[Document, Iterable[Document], None] = None,
        split: int = 0,
        sort: bool = False,
    ) -> List[List[Candidate]]:
        """Return a list of lists of the candidates associated with this extractor.

        Each list of the return will contain the candidates for one of the
        candidate classes associated with the CandidateExtractor.

        :param docs: If provided, return candidates from these documents from
            all splits.
        :param split: If docs is None, then return all the candidates from this
            split.
        :param sort: If sort is True, then return all candidates sorted by stable_id.
        :return: Candidates for each candidate_class.
        """
        result = []
        if docs:
            docs = docs if isinstance(docs, Iterable) else [docs]
            # Get cands from all splits
            for candidate_class in self.candidate_classes:
                cands = (
                    self.session.query(candidate_class)
                    .filter(candidate_class.document_id.in_([doc.id for doc in docs]))
                    .order_by(candidate_class.id)
                    .all()
                )
                if sort:
                    cands = sorted(
                        cands,
                        key=lambda x: "_".join(
                            [x[i].context.get_stable_id() for i in range(len(x))]
                        ),
                    )
                result.append(cands)
        else:
            for candidate_class in self.candidate_classes:
                # Filter by candidate_ids in a particular split
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
                if sort:
                    cands = sorted(
                        cands,
                        key=lambda x: "_".join(
                            [x[i].context.get_stable_id() for i in range(len(x))]
                        ),
                    )
                result.append(cands)
        return result


# Type alias for throttler
Throttler = Callable[[Tuple[Mention, ...]], bool]


class CandidateExtractorUDF(UDF):
    """UDF for performing candidate extraction."""

    def __init__(
        self,
        candidate_classes: Union[Type[Candidate], List[Type[Candidate]]],
        throttlers: Union[Throttler, List[Throttler]],
        self_relations: bool,
        nested_relations: bool,
        symmetric_relations: bool,
        **kwargs: Any,
    ) -> None:
        """Initialize the CandidateExtractorUDF."""
        self.candidate_classes = (
            candidate_classes
            if isinstance(candidate_classes, (list, tuple))
            else [candidate_classes]
        )
        self.throttlers = (
            throttlers if isinstance(throttlers, (list, tuple)) else [throttlers]
        )
        self.nested_relations = nested_relations
        self.self_relations = self_relations
        self.symmetric_relations = symmetric_relations
        self.arities = [len(cclass.__argnames__) for cclass in self.candidate_classes]

        super().__init__(**kwargs)

    def apply(  # type: ignore
        self, doc: Document, split: int, **kwargs: Any
    ) -> Document:
        """Extract candidates from the given Context.

        :param doc: A document to process.
        :param split: Which split to use.
        """
        logger.debug(f"Document: {doc}")
        # Iterate over each candidate class
        for i, candidate_class in enumerate(self.candidate_classes):
            logger.debug(f"  Relation: {candidate_class.__name__}")
            # Generates and persists candidates
            candidate_args = {"split": split}
            candidate_args["document"] = doc
            cands = product(
                *[
                    enumerate(
                        # a list of mentions for each mention subclass within a doc
                        getattr(doc, mention.__tablename__ + "s")
                        + ([None] if nullable else [])
                    )
                    for mention, nullable in zip(
                        candidate_class.mentions, candidate_class.nullables
                    )
                ]
            )
            # Get a set of stable_ids of candidates.
            set_of_stable_ids = get_set_of_stable_ids(doc, candidate_class)

            for cand in cands:

                # Apply throttler if one was given.
                # Accepts a tuple of Mention objects
                # (throttler returns whether or not proposed candidate
                # passes throttling condition)
                if self.throttlers[i]:
                    if not self.throttlers[i](
                        tuple(cand[j][1] for j in range(self.arities[i]))
                    ):
                        continue

                # TODO: Make this work for higher-order relations
                if self.arities[i] == 2:
                    ai, a = (cand[0][0], cand[0][1].context if cand[0][1] else None)
                    bi, b = (cand[1][0], cand[1][1].context if cand[1][1] else None)

                    # Check for self-joins, "nested" joins (joins from context to
                    # its subcontext), and flipped duplicate "symmetric" relations
                    if not self.self_relations and a == b:
                        logger.debug(f"Skipping self-joined candidate {cand}")
                        continue
                    # Skip the check if either is None as None is not iterable.
                    if not self.nested_relations and (a and b) and (a in b or b in a):
                        logger.debug(f"Skipping nested candidate {cand}")
                        continue
                    if not self.symmetric_relations and ai > bi:
                        logger.debug(f"Skipping symmetric candidate {cand}")
                        continue

                # Assemble candidate arguments
                for j, arg_name in enumerate(candidate_class.__argnames__):
                    candidate_args[arg_name] = cand[j][1]

                stable_ids = tuple(
                    cand[j][1].context.get_stable_id() if cand[j][1] else None
                    for j in range(self.arities[i])
                )
                # Skip if this (temporary) candidate is used by this candidate class.
                if (
                    hasattr(doc, candidate_class.__tablename__ + "s")
                    and stable_ids in set_of_stable_ids
                ):
                    continue

                # Add Candidate to session
                candidate_class(**candidate_args)
        return doc
