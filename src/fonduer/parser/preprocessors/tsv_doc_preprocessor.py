import codecs

from fonduer.parser.models import Document
from fonduer.parser.preprocessors.doc_preprocessor import DocPreprocessor


class TSVDocPreprocessor(DocPreprocessor):
    """Simple parsing of TSV file with one (doc_name <tab> doc_text) per line"""

    def parse_file(self, fp, file_name):
        with codecs.open(fp, encoding=self.encoding) as tsv:
            for line in tsv:
                (doc_name, doc_text) = line.split("\t")
                stable_id = self.get_stable_id(doc_name)
                doc = Document(
                    name=doc_name, stable_id=stable_id, meta={"file_name": file_name}
                )
                yield doc, doc_text
