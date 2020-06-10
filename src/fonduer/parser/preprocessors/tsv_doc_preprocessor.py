"""Fonduer TSV document preprocessor."""
import codecs
import sys
from typing import Iterator

from fonduer.parser.models import Document
from fonduer.parser.preprocessors.doc_preprocessor import DocPreprocessor
from fonduer.utils.utils_parser import build_node


class TSVDocPreprocessor(DocPreprocessor):
    """A ``Document`` generator for TSV files.

    It treats each line in the input file as a ``Document``.
    The TSV file should have one (doc_name <tab> doc_text) per line.
    """

    def __init__(
        self,
        path: str,
        encoding: str = "utf-8",
        max_docs: int = sys.maxsize,
        header: bool = False,
    ) -> None:
        """Initialize TSV DocPreprocessor.

        :param path: a path to file or directory, or a glob pattern. The basename
            (as returned by ``os.path.basename``) should be unique among all files.
        :param encoding: file encoding to use (e.g. "utf-8").
        :param max_docs: the maximum number of ``Documents`` to produce.
        :param header: if the TSV file contain header or not. default = False
        :return: A generator of ``Documents``.
        """
        super().__init__(path, encoding, max_docs)
        self.header = header

    def _parse_file(self, fp: str, file_name: str) -> Iterator[Document]:
        with codecs.open(fp, encoding=self.encoding) as tsv:
            if self.header:
                tsv.readline()
            for line in tsv:
                (doc_name, doc_text) = line.split("\t")
                stable_id = self._get_stable_id(doc_name)
                text = build_node("doc", None, build_node("text", None, doc_text))
                yield Document(
                    name=doc_name,
                    stable_id=stable_id,
                    text=text,
                    meta={"file_name": file_name},
                )

    def __len__(self) -> int:
        """Provide a len attribute based on max_docs and number of rows in files."""
        cnt_docs = 0
        for fp in self.all_files:
            with codecs.open(fp, encoding=self.encoding) as tsv:
                num_lines = sum(1 for line in tsv)
                cnt_docs += num_lines - 1 if self.header else num_lines
            if cnt_docs > self.max_docs:
                break
        num_docs = min(cnt_docs, self.max_docs)
        return num_docs

    def _can_read(self, fpath: str) -> bool:
        return fpath.lower().endswith(".tsv")
