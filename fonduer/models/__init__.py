from fonduer.snorkel.models.candidate import candidate_subclass
from fonduer.snorkel.models.context import Document
from fonduer.snorkel.models.meta import SnorkelSession, SnorkelBase, snorkel_engine

from fonduer.models.context import Cell
from fonduer.models.context import Figure
from fonduer.models.context import FigureCaption
from fonduer.models.context import Header
from fonduer.models.context import Image
from fonduer.models.context import ImplicitSpan
from fonduer.models.context import Para
from fonduer.models.context import Phrase
from fonduer.models.context import RefList
from fonduer.models.context import Section
from fonduer.models.context import Table
from fonduer.models.context import TableCaption
from fonduer.models.context import TemporaryImage
from fonduer.models.context import TemporaryImplicitSpan
from fonduer.models.context import Webpage

# Use sqlalchemy to create tables for the new context types used by Fonduer
SnorkelBase.metadata.create_all(snorkel_engine)
