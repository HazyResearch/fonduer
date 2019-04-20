import logging

from sqlalchemy import Column, String
from sqlalchemy.orm import relationship

from fonduer.meta import Meta

logger = logging.getLogger(__name__)


class Entity(Meta.Base):

    __tablename__ = "entity"

    #: ``Entity``'s textual representation and the unique id for the table.
    id = Column(String, primary_key=True)

    #: The type for the ``Entity``, which corresponds to the names the user
    #: gives to the corresponding mention_subclass.
    type = Column(String, nullable=False)

    mentions = relationship("Mention", backref="entity")
