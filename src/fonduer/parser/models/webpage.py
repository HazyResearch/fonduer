from sqlalchemy import Column, ForeignKey, Integer, String

from fonduer.parser.models.document import Document


class Webpage(Document):
    """A Webpage document context enhanced with additional metadata."""

    __tablename__ = "webpage"

    #: The unique id of the ``Webpage``.
    id = Column(
        Integer, ForeignKey("document.id", ondelete="CASCADE"), primary_key=True
    )
    #: The URL of the ``Webpage``.
    url = Column(String)
    #: The host of the ``Webpage``.
    host = Column(String)

    #: The type of the ``Webpage``.
    page_type = Column(String)

    #: The raw content of the ``Webpage``.
    raw_content = Column(String)

    #: The timestamp of when the ``Webpage`` was crawled.
    crawltime = Column(String)
    all = Column(String)

    # Polymorphism information for SQLAlchemy
    __mapper_args__ = {"polymorphic_identity": "webpage"}

    # Rest of class definition here
    def __repr__(self):
        return "Webpage(id: {}..., url: {}...)".format(self.name[:10], self.url[8:23])
