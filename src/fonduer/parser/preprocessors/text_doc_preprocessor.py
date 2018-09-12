import codecs
import os

from fonduer.parser.models import Document
from fonduer.parser.preprocessors.doc_preprocessor import DocPreprocessor


class TextDocPreprocessor(DocPreprocessor):
    """A generator for simple parsing of raw text files.

    Assumes one document per file.
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
