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
        """Parse visual information and link them with given sentences.

        :param document_name: the document name.
        :param sentences: sentences to be linked with visual information.
        :yield: sentences with visual information.
        """
        pass

    @abstractmethod
    def is_parsable(self, document_name: str) -> bool:
        """Check if visual information can be parsed.

        :param document_name: the document name.
        :return: Whether visual information is parsable.
        """
        pass
