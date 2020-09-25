"""Abstract visual parser."""
from abc import ABC, abstractmethod
from typing import Iterable, Iterator

from fonduer.parser.models import Sentence


class VisualParser(ABC):
    """Abstract visual parer."""

    @abstractmethod
    def parse(
        self,
        document_name: str,
        sentences: Iterable[Sentence],
    ) -> Iterator[Sentence]:
        pass

    @abstractmethod
    def is_parsable(self, document_name: str) -> bool:
        pass
