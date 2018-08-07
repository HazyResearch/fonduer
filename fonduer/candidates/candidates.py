import logging
from builtins import range
from itertools import product

from sqlalchemy.sql import select

from fonduer.candidates.models import Candidate
from fonduer.utils.udf import UDF, UDFRunner

logger = logging.getLogger(__name__)


class CandidateExtractor(UDFRunner):
    """An operator to extract Candidate objects from a Context.

    :param candidate_class: The type of relation to extract, defined using
        :func:`fonduer.candidates.candidate_subclass.
    :param cspaces: one or list of :class:`CandidateSpace` objects, one for
        each relation argument. Defines space of Contexts to consider
    :param matchers: one or list of :class:`fonduer.matchers.Matcher` objects,
        one for each relation argument. Only tuples of Contexts for which each
        element is accepted by the corresponding Matcher will be returned as
        Candidates
    :param candidate_throttler: an optional function for filtering out candidates
        which returns a Boolean expressing whether or not the candidate should
        be instantiated.
    :param self_relations: Boolean indicating whether to extract Candidates
        that relate the same context. Only applies to binary relations. Default
        is False.
    :param nested_relations: Boolean indicating whether to extract Candidates
        that relate one Context with another that contains it. Only applies to
        binary relations. Default is False.
    :param symmetric_relations: Boolean indicating whether to extract symmetric
        Candidates, i.e., rel(A,B) and rel(B,A), where A and B are Contexts.
        Only applies to binary relations. Default is True.
    """

    def __init__(
        self,
        candidate_classes,
        candidate_throttlers=None,
        self_relations=False,
        nested_relations=False,
        symmetric_relations=True,
    ):
        """Initialize the CandidateExtractor."""
        super(CandidateExtractor, self).__init__(
            CandidateExtractorUDF,
            candidate_classes=candidate_classes,
            candidate_throttlers=candidate_throttlers,
            self_relations=self_relations,
            nested_relations=nested_relations,
            symmetric_relations=symmetric_relations,
        )
        # Check that arity is sensible
        if len(candidate_classes) < len(candidate_throttlers):
            raise ValueError("Provided more throttlers than candidate classes.")

        self.candidate_classes = candidate_classes

    def apply(self, xs, split=0, **kwargs):
        """Call the CandidateExtractorUDF."""
        super(CandidateExtractor, self).apply(xs, split=split, **kwargs)

    def clear(self, session, split, **kwargs):
        """Delete Candidates of each cass from given split the database."""
        for candidate_class in self.candidate_classes:
            logger.info("Clearing {}".format(candidate_class.__tablename__))
            session.query(Candidate).filter(
                Candidate.type == candidate_class.__tablename__
            ).filter(Candidate.split == split).delete()

    def clear_all(self, session, split, **kwargs):
        """Delete all Candidates from given split the database."""
        logger.info("Clearing ALL Candidates.")
        session.query(Candidate).filter(Candidate.split == split).delete()


class CandidateExtractorUDF(UDF):
    """UDF for performing candidate extraction."""

    def __init__(
        self,
        candidate_classes,
        candidate_throttlers,
        self_relations,
        nested_relations,
        symmetric_relations,
        **kwargs
    ):
        """Initialize the CandidateExtractorUDF."""
        self.candidate_classes = (
            candidate_classes
            if type(candidate_classes) in [list, tuple]
            else [candidate_classes]
        )
        self.candidate_throttlers = (
            candidate_throttlers
            if type(candidate_throttlers) in [list, tuple]
            else [candidate_throttlers]
        )
        self.nested_relations = nested_relations
        self.self_relations = self_relations
        self.symmetric_relations = symmetric_relations
        self.arities = [len(cclass.__argnames__) for cclass in self.candidate_classes]

        super(CandidateExtractorUDF, self).__init__(**kwargs)

    def apply(self, context, clear, split, **kwargs):
        """Extract candidates from the given Context.

        :param context: A document to process.
        :param clear: Whether or not to clear the existing database entries.
        :param split: Which split to use.
        """
        logger.info("Document: {}".format(context))
        # Iterate over each candidate class
        for i, candidate_class in enumerate(self.candidate_classes):
            logger.info("  Relation: {}".format(candidate_class.__name__))
            # Generates and persists candidates
            candidate_args = {"split": split}
            candidate_args["document_id"] = context.id
            cands = product(
                *[
                    enumerate(
                        self.session.query(mention)
                        .filter(mention.document_id == context.id)
                        .order_by(mention.id)
                        .all()
                    )
                    for mention in candidate_class.mentions
                ]
            )
            for k, cand in enumerate(cands):

                # Apply candidate_throttler if one was given.
                # Accepts a tuple of Context objects (e.g., (Span, Span))
                # (candidate_throttler returns whether or not proposed candidate
                # passes throttling condition)
                if self.candidate_throttlers[i]:
                    if not self.candidate_throttlers[i](
                        tuple(cand[j][1].span for j in range(self.arities[i]))
                    ):
                        continue

                # TODO: Make this work for higher-order relations
                if self.arities[i] == 2:
                    ai, a = (cand[0][0], cand[0][1].span)
                    bi, b = (cand[1][0], cand[1][1].span)

                    # Check for self-joins, "nested" joins (joins from span to
                    # its subspan), and flipped duplicate "symmetric" relations
                    if not self.self_relations and a == b:
                        logger.info("Skipping self-joined candidate {}".format(cand))
                        continue
                    if not self.nested_relations and (a in b or b in a):
                        logger.info("Skipping nested candidate {}".format(cand))
                        continue
                    if not self.symmetric_relations and ai > bi:
                        logger.info("Skipping symmetric candidate {}".format(cand))
                        continue

                # Assemble candidate arguments
                for j, arg_name in enumerate(candidate_class.__argnames__):
                    candidate_args[arg_name + "_id"] = cand[j][1].id

                # Checking for existence
                if not clear:
                    q = select([candidate_class.id])
                    for key, value in list(candidate_args.items()):
                        q = q.where(getattr(candidate_class, key) == value)
                    candidate_id = self.session.execute(q).first()
                    if candidate_id is not None:
                        continue

                # Add Candidate to session
                yield candidate_class(**candidate_args)
