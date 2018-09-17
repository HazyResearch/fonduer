import codecs

from fonduer.parser.models import Document
from fonduer.parser.preprocessors.doc_preprocessor import DocPreprocessor


class TSVDocPreprocessor(DocPreprocessor):
    """A generator which processes a TSV file or directory of TSV files into
    a set of Document objects.

    The TSV file should have one (doc_name <tab> doc_test) per line.

    :param encoding: file encoding to use (e.g. "utf-8").
    :type encoding: str
    :param path: filesystem path to file or directory to parse.
    :type path: str
    :param max_docs: the maximum number of ``Documents`` to produce.
    :type max_docs: int
    :rtype: A generator of ``Documents``.
    """

    def _parse_file(self, fp, file_name):
        with codecs.open(fp, encoding=self.encoding) as tsv:
            for line in tsv:
                (doc_name, doc_text) = line.split("\t")
                stable_id = self._get_stable_id(doc_name)
                doc = Document(
                    name=doc_name,
                    stable_id=stable_id,
                    meta={"file_name": file_name},
                    text=doc_text,
                )
                yield doc
