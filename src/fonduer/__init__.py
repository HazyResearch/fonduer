import logging

from fonduer._version import __version__
from fonduer.meta import Meta

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["__version__", "Meta"]
