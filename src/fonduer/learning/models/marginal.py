"""Fonduer learning marginal model."""
from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, UniqueConstraint

from fonduer.meta import Meta


class Marginal(Meta.Base):
    """
    A marginal probability corresponding to a (Candidate, value) pair.

    Represents:

        P(candidate = value) = probability

    @training: If True, this is a training marginal; otherwise is end prediction
    """

    __tablename__ = "marginal"
    id = Column(Integer, primary_key=True)
    candidate_id = Column(Integer, ForeignKey("candidate.id", ondelete="CASCADE"))
    training = Column(Boolean, default=True)
    value = Column(Integer, nullable=False, default=1)
    probability = Column(Float, nullable=False, default=0.0)

    __table_args__ = (UniqueConstraint(candidate_id, training, value),)

    def __repr__(self) -> str:
        """Represent the marginal as a string."""
        label = "Training" if self.training else "Predicted"
        return (
            f"<"
            f"{label} "
            f"Marginal: P({self.candidate_id} == {self.value}) = {self.probability}"
            f">"
        )
