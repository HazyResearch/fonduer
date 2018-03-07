from fonduer.snorkel.models.candidate import candidate_subclass
from fonduer.snorkel.models.context import Document
from fonduer.snorkel.models.meta import SnorkelSession, SnorkelBase, snorkel_engine

from fonduer.models.context import Webpage, Table, Cell, Phrase, Figure, TemporaryImplicitSpan, ImplicitSpan, TemporaryImage, Image

# Use sqlalchemy to create tables for the new context types used by Fonduer
SnorkelBase.metadata.create_all(snorkel_engine)
