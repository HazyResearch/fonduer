import codecs
import os

from fonduer.parser.models import Document
from fonduer.parser.preprocessors.doc_preprocessor import DocPreprocessor


class TextDocPreprocessor(DocPreprocessor):
    """Simple parsing of raw text files, assuming one document per file"""

    def parse_file(self, fp, file_name):
        with codecs.open(fp, encoding=self.encoding) as f:
            name = os.path.basename(fp).rsplit(".", 1)[0]
            stable_id = self.get_stable_id(name)
            doc = Document(
                name=name, stable_id=stable_id, meta={"file_name": file_name}
            )
            yield doc, f.read()
