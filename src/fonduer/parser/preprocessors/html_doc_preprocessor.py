import codecs
import os

from bs4 import BeautifulSoup

from fonduer.parser.models import Document
from fonduer.parser.preprocessors.doc_preprocessor import DocPreprocessor


class HTMLDocPreprocessor(DocPreprocessor):
    """Simple parsing of files into html documents"""

    def parse_file(self, fp, file_name):
        with codecs.open(fp, encoding=self.encoding) as f:
            soup = BeautifulSoup(f, "lxml")
            for text in soup.find_all("html"):
                name = os.path.basename(fp)[: os.path.basename(fp).rfind(".")]
                stable_id = self.get_stable_id(name)
                yield Document(
                    name=name,
                    stable_id=stable_id,
                    text=str(text),
                    meta={"file_name": file_name},
                ), str(text)

    def _can_read(self, fpath):
        return fpath.endswith("html")  # includes both .html and .xhtml
