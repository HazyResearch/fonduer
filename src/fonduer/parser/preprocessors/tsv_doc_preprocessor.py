import codecs

from fonduer.parser.models import Document
from fonduer.parser.preprocessors.doc_preprocessor import DocPreprocessor
from fonduer.utils.utils_parser import build_node


class TSVDocPreprocessor(DocPreprocessor):
    """A generator which processes a TSV file or directory of TSV files into
    a set of Document objects.

    The TSV file should have one (doc_name <tab> doc_text) per line.

    :param path: filesystem path to file or directory to parse.
    :type path: str
    :param encoding: file encoding to use (e.g. "utf-8").
    :type encoding: str
    :param max_docs: the maximum number of ``Documents`` to produce.
    :type max_docs: int
    :param header: if the TSV file contain header or not. default = False
    :type header: bool
    :rtype: A generator of ``Documents``.
    """

    def __init__(self, path, encoding="utf-8", max_docs=float("inf"), header=False):
        super(TSVDocPreprocessor, self).__init__(path, encoding, max_docs)
        self.header = header

    def _parse_file(self, fp, file_name):
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

    def __len__(self):
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

    def _can_read(self, fpath):
        return fpath.lower().endswith(".tsv")
