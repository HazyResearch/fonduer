import logging

from sqlalchemy import Column, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import backref, relationship

from fonduer.meta import Meta

_meta = Meta.init()
logger = logging.getLogger(__name__)


class Label(_meta.Base):
    """
    A Label associated with a Candidate.
    """

    __tablename__ = "label"
    id = Column(Integer, primary_key=True)
    value = Column(Integer, nullable=False)
    candidate_id = Column(Integer, ForeignKey("candidate.id", ondelete="CASCADE"))
    candidate = relationship(
        "Candidate",
        backref=backref("labels", order_by=value, cascade="all, delete-orphan"),
        foreign_keys=candidate_id,
    )

    __mapper_args__ = {"polymorphic_identity": "label"}

    __table_args__ = (UniqueConstraint(candidate_id, value),)

    def __repr__(self):
        return "Label(Cand: {}, Value: {})".format(self.candidate, self.value)

    def __gt__(self, other):
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()
