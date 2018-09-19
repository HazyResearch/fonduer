from sqlalchemy import Column, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import backref, relationship

from fonduer.candidates.models.temporarycontext import TemporaryContext
from fonduer.parser.models.context import Context


class TemporaryImage(TemporaryContext):
    """The TemporaryContext version of Image"""

    def __init__(self, figure):
        super(TemporaryImage, self).__init__()
        self.figure = figure  # The figure Context

    def __len__(self):
        return 1

    def __eq__(self, other):
        try:
            return (
                self.figure.position == other.figure.position
                and self.figure.url == other.figure.url
            )
        except AttributeError:
            return False

    def __ne__(self, other):
        try:
            return (
                self.figure.position != other.figure.position
                or self.figure.url != other.figure.url
            )
        except AttributeError:
            return True

    def __contains__(self, other_span):
        return self.__eq__(other_span)

    def __hash__(self):
        return hash(self.figure)

    def get_stable_id(self):
        """Return a stable id for the ``Image``."""
        return "%s::%s:%s" % (
            self.figure.document.id,
            self._get_polymorphic_identity(),
            self.figure.position,
        )

    def _get_table(self):
        return Image

    def _get_polymorphic_identity(self):
        return "image"

    def _get_insert_args(self):
        return {
            "document_id": self.figure.document.id,
            "position": self.figure.position,
            "url": self.figure.url,
        }

    def __repr__(self):
        return "{}(document={}, position={}, url={})".format(
            self.__class__.__name__,
            self.figure.document.name,
            self.figure.position,
            self.figure.url,
        )

    def _get_instance(self, **kwargs):
        return TemporaryImage(**kwargs)


class Image(Context, TemporaryImage):
    """A figure-based ``Candidate``.

    Note that this image should not be confused with the ``Figure`` context of
    the document model.
    """

    __tablename__ = "image"

    #: The unique id of the ``Image``.
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)

    #: The id of the parent ``Document``.
    document_id = Column(Integer, ForeignKey("document.id", ondelete="CASCADE"))

    #: The position of the ``Image`` in the ``Document``.
    position = Column(Integer, nullable=False)

    #: The parent ``Document``.
    document = relationship(
        "Document",
        backref=backref("images", order_by=position, cascade="all, delete-orphan"),
        foreign_keys=document_id,
    )

    #: The url of the ``Image``.
    url = Column(String)

    __table_args__ = (UniqueConstraint(document_id, position),)

    __mapper_args__ = {"polymorphic_identity": "image"}

    def __repr__(self):
        return "Image(Doc: {}, Position: {}, Url: {})".format(
            self.document.name, self.position, self.url
        )

    def __gt__(self, other):
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()
