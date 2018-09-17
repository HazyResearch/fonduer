import codecs
import os

from fonduer.parser.models import Document
from fonduer.parser.preprocessors.doc_preprocessor import DocPreprocessor


class TextDocPreprocessor(DocPreprocessor):
    """A generator which processes a text file or directory of text files into
    a set of Document objects.

    Assumes one document per file.

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
            name = os.path.basename(fp).rsplit(".", 1)[0]
            stable_id = self._get_stable_id(name)
            doc = Document(
                name=name,
                stable_id=stable_id,
                meta={"file_name": file_name},
                text=f.read(),
            )
            yield doc
