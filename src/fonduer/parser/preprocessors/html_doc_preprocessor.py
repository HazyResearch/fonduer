import codecs
import os

from bs4 import BeautifulSoup

from fonduer.parser.models import Document
from fonduer.parser.preprocessors.doc_preprocessor import DocPreprocessor


class HTMLDocPreprocessor(DocPreprocessor):
    """A generator which processes an HTML file or directory of HTML files into
    a set of Document objects.

    :param encoding: file encoding to use (e.g. "utf-8").
    :type encoding: str
    :param path: filesystem path to file or directory to parse.
    :type path: str
    :param max_docs: the maximum number of ``Documents`` to produce.
    :type max_docs: int
    :rtype: A generator of ``Documents``.
    """

    def _parse_file(self, fp, file_name):
        with codecs.open(fp, encoding=self.encoding) as f:
            soup = BeautifulSoup(f, "lxml")
            all_html_elements = soup.find_all("html")
            if len(all_html_elements) != 1:
                raise NotImplementedError(
                    "Expecting exactly one html element per html file: {}".format(
                        file_name
                    )
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

    def __len__(self):
        """Provide a len attribute based on max_docs and number of files in folder."""
        num_docs = min(len(self.all_files), self.max_docs)
        return num_docs

    def _can_read(self, fpath):
        return fpath.lower().endswith("html")  # includes both .html and .xhtml
