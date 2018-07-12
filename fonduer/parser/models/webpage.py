from sqlalchemy import Column, ForeignKey, Integer, String

from fonduer.parser.models.document import Document


class Webpage(Document):
    """
    Declares name for storage table.
    """

    __tablename__ = "webpage"
    id = Column(
        Integer, ForeignKey("document.id", ondelete="CASCADE"), primary_key=True
    )
    # Connects NewType records to generic Context records
    url = Column(String)
    host = Column(String)
    page_type = Column(String)
    raw_content = Column(String)
    crawltime = Column(String)
    all = Column(String)

    # Polymorphism information for SQLAlchemy
    __mapper_args__ = {"polymorphic_identity": "webpage"}

    # Rest of class definition here
    def __repr__(self):
        return "Webpage(id: {}..., url: {}...)".format(self.name[:10], self.url[8:23])
