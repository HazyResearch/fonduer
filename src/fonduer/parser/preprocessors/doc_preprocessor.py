"""Fonduer document preprocessor."""
import glob
import os
import sys
from typing import Iterator, List

from fonduer.parser.models.document import Document


class DocPreprocessor(object):
    """An abstract class of a ``Document`` generator.

    Unless otherwise stated by a subclass, it's assumed that there is one ``Document``
    per file.
    """

    def __init__(
        self, path: str, encoding: str = "utf-8", max_docs: int = sys.maxsize
    ) -> None:
        """Initialize DocPreprocessor.

        :param path: a path to file or directory, or a glob pattern. The basename
            (as returned by ``os.path.basename``) should be unique among all files.
        :param encoding: file encoding to use, defaults to "utf-8".
        :param max_docs: the maximum number of ``Documents`` to produce,
            defaults to sys.maxsize.
        :return: A generator of ``Documents``.
        """
        self.path = path
        self.encoding = encoding
        self.max_docs = max_docs
        self.all_files = self._get_files(self.path)

    def _generate(self) -> Iterator[Document]:
        """Parse a file or directory of files into a set of ``Document`` objects."""
        doc_count = 0
        for fp in self.all_files:
            for doc in self._get_docs_for_path(fp):
                yield doc
                doc_count += 1
                if doc_count >= self.max_docs:
                    return

    def __len__(self) -> int:
        """Get total number of documents."""
        raise NotImplementedError(
            "One generic file can yield more than one Document object, "
            "so length can not be yielded before we process all files"
        )

    def __iter__(self) -> Iterator[Document]:
        """Get the generator of documents."""
        return self._generate()

    def _get_docs_for_path(self, fp: str) -> Iterator[Document]:
        file_name = os.path.basename(fp)
        if self._can_read(file_name):
            for doc in self._parse_file(fp, file_name):
                yield doc

    def _get_stable_id(self, doc_id: str) -> str:
        return f"{doc_id}::document:0:0"

    def _parse_file(self, fp: str, file_name: str) -> Iterator[Document]:
        raise NotImplementedError()

    def _can_read(self, fpath: str) -> bool:
        return not fpath.startswith(".")

    def _get_files(self, path: str) -> List[str]:
        if os.path.isfile(path):
            fpaths = [path]
        elif os.path.isdir(path):
            fpaths = [os.path.join(path, f) for f in os.listdir(path)]
        else:
            fpaths = glob.glob(path)
        fpaths = [x for x in fpaths if self._can_read(x)]
        if len(fpaths) > 0:
            return sorted(fpaths)
        else:
            raise IOError(
                f"No readable file found at {path}. Check if {path} exists "
                "and a filename ends with an allowable extension (e.g., .html)"
            )
