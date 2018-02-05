from snorkel.models.candidate import candidate_subclass
from snorkel.models.context import Document
from snorkel.models.meta import SnorkelSession, SnorkelBase, snorkel_engine

from fonduer.models.context import Webpage, Table, Cell, Phrase, Figure, TemporaryImplicitSpan, ImplicitSpan, TemporaryImage, Image

# Use sqlalchemy to create tables for the new context types used by Fonduer
SnorkelBase.metadata.create_all(snorkel_engine)
