import logging

from sqlalchemy import Column, String
from sqlalchemy.orm import relationship

from fonduer.meta import Meta

logger = logging.getLogger(__name__)


class Entity(Meta.Base):

    __tablename__ = "entity"

    #: ``Entity``'s textual representation and the unique id for the table.
    id = Column(String, primary_key=True)

    mentions = relationship("Mention", backref="entity")

    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
        return self.id > other.id
