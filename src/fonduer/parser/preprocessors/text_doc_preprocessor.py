"""Fonduer text document preprocessor."""
import codecs
import os
from typing import Iterator

from fonduer.parser.models import Document
from fonduer.parser.preprocessors.doc_preprocessor import DocPreprocessor
from fonduer.utils.utils_parser import build_node


class TextDocPreprocessor(DocPreprocessor):
    """A ``Document`` generator for plain text files."""

    def _parse_file(self, fp: str, file_name: str) -> Iterator[Document]:
        with codecs.open(fp, encoding=self.encoding) as f:
            name = os.path.basename(fp).rsplit(".", 1)[0]
            stable_id = self._get_stable_id(name)
            text = build_node("doc", None, build_node("text", None, f.read().strip()))
            yield Document(
                name=name, stable_id=stable_id, text=text, meta={"file_name": file_name}
            )

    def __len__(self) -> int:
        """Provide a len attribute based on max_docs and number of files in folder."""
        num_docs = min(len(self.all_files), self.max_docs)
        return num_docs
