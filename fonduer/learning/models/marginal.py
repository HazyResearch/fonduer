from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, UniqueConstraint

from fonduer.meta import Meta

_meta = Meta.init()


class Marginal(_meta.Base):
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

    def __repr__(self):
        label = "Training" if self.training else "Predicted"
        return "<%s Marginal: P(%s == %s) = %s>" % (
            label,
            self.candidate_id,
            self.value,
            self.probability,
        )
