"""Fonduer HTML document preprocessor."""
import codecs
import os
from typing import Iterator

from bs4 import BeautifulSoup

from fonduer.parser.models import Document
from fonduer.parser.preprocessors.doc_preprocessor import DocPreprocessor


class HTMLDocPreprocessor(DocPreprocessor):
    """A ``Document`` generator for HTML files."""

    def _parse_file(self, fp: str, file_name: str) -> Iterator[Document]:
        with codecs.open(fp, encoding=self.encoding) as f:
            soup = BeautifulSoup(f, "lxml")
            all_html_elements = soup.find_all("html")
            if len(all_html_elements) != 1:
                raise NotImplementedError(
                    f"Expecting exactly one html element per html file: {file_name}"
                )
            text = all_html_elements[0]
            name = os.path.basename(fp)[: os.path.basename(fp).rfind(".")]
            stable_id = self._get_stable_id(name)
            yield Document(
                name=name,
                stable_id=stable_id,
                text=str(text),
                meta={"file_name": file_name},
            )

    def __len__(self) -> int:
        """Provide a len attribute based on max_docs and number of files in folder."""
        num_docs = min(len(self.all_files), self.max_docs)
        return num_docs

    def _can_read(self, fpath: str) -> bool:
        return fpath.lower().endswith("html")  # includes both .html and .xhtml
