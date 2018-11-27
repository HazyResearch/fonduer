from sqlalchemy import Column, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import relationship

from fonduer.candidates.models.temporary_context import TemporaryContext
from fonduer.parser.models.context import Context


class TemporaryDocumentMention(TemporaryContext):
    """The TemporaryContext version of DocumentMention."""

    def __init__(self, document):
        super(TemporaryDocumentMention, self).__init__()
        self.document = document  # The document Context

    def __len__(self):
        return 1

    def __eq__(self, other):
        try:
            return self.document == other.document
        except AttributeError:
            return False

    def __ne__(self, other):
        try:
            return self.document != other.document
        except AttributeError:
            return True

    def __contains__(self, other_document):
        return self.__eq__(other_document)

    def __hash__(self):
        return hash(self.document)

    def get_stable_id(self):
        """Return a stable id for the ``DocumentMention``."""
        return "%s::%s" % (self.document.name, self._get_polymorphic_identity())

    def _get_table(self):
        return DocumentMention

    def _get_polymorphic_identity(self):
        return "document_mention"

    def _get_insert_args(self):
        return {"document_id": self.document.id}

    def __repr__(self):
        return "{}(document={})".format(self.__class__.__name__, self.document.name)

    def _get_instance(self, **kwargs):
        return TemporaryDocumentMention(**kwargs)


class DocumentMention(Context, TemporaryDocumentMention):
    """A document ``Mention``."""

    __tablename__ = "document_mention"

    #: The unique id of the ``DocumentMention``.
    id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"), primary_key=True)

    #: The id of the parent ``Document``.
    document_id = Column(Integer, ForeignKey("context.id", ondelete="CASCADE"))

    #: The parent ``Document``.
    document = relationship("Context", foreign_keys=document_id)

    __table_args__ = (UniqueConstraint(document_id),)

    __mapper_args__ = {
        "polymorphic_identity": "document_mention",
        "inherit_condition": (id == Context.id),
    }

    def __gt__(self, other):
        # Allow sorting by comparing the string representations of each
        return self.__repr__() > other.__repr__()
