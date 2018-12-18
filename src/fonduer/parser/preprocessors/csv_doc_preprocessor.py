import codecs
import csv
import os

from fonduer.parser.models import Document
from fonduer.parser.preprocessors.doc_preprocessor import DocPreprocessor
from fonduer.utils.utils_parser import build_node, column_constructor


class CSVDocPreprocessor(DocPreprocessor):
    """A generator which processes a CSV file or directory of CSV files into
    a set of Document objects. It treats each line in the input file is a document.
    This ``DocPreprocessor`` assumes that each column is one section and content in
    each column as one paragraph as defalt. However, if the column is complex, an
    advanced parser may be used by specifying ``parser_rule`` parameter, e,g.,
    specify keywords as delimiters for paragraph in a dict format where key is the
    column inedx and value is the keyword list.

    :param path: filesystem path to file or directory to parse.
    :type path: str
    :param encoding: file encoding to use (e.g. "utf-8").
    :type encoding: str
    :param max_docs: the maximum number of ``Documents`` to produce.
    :type max_docs: int
    :param header: if the CSV file contain header or not, if yes, the header
        will be used as Section name. default = False
    :type header: bool
    :param delim: delimiter to be used to separate columns when file has
        more than one column. It is active only when ``column is not
        None``. default=','
    :type delim: int
    :param parser_rule: The parser rule to be used to parse the specific column.
        default = None
    :rtype: A generator of ``Documents``.
    """

    def __init__(
        self,
        path,
        encoding="utf-8",
        max_docs=float("inf"),
        header=False,
        delim=",",
        parser_rule=None,
    ):
        super(CSVDocPreprocessor, self).__init__(path, encoding, max_docs)
        self.header = header
        self.delim = delim
        self.parser_rule = parser_rule
        self.n_parsed = 0

    def _parse_file(self, fp, file_name):
        name = os.path.basename(fp)[: os.path.basename(fp).rfind(".")]
        with codecs.open(fp, encoding=self.encoding) as f:
            reader = csv.reader(f)

            # Load CSV header
            header_names = None
            if self.header:
                header_names = next(reader)

            # Load document per row
            for i, row in enumerate(reader):
                if self.n_parsed == self.max_docs:
                    break
                sections = []
                for j, content in enumerate(row):
                    rule = (
                        self.parser_rule[j]
                        if self.parser_rule is not None and j in self.parser_rule
                        else column_constructor
                    )
                    content_header = (
                        header_names[j] if header_names is not None else None
                    )
                    context = [build_node(t, n, c) for t, n, c in rule(content)]
                    sections.append(
                        build_node("section", content_header, "".join(context))
                    )

                text = build_node("doc", None, "".join(sections))
                doc_name = name + ":" + str(i)
                stable_id = self._get_stable_id(doc_name)

                yield Document(
                    name=doc_name,
                    stable_id=stable_id,
                    text=text,
                    meta={"file_name": file_name},
                )
                self.n_parsed += 1

    def __len__(self):
        """Provide a len attribute based on max_docs and number of files in folder."""
        num_docs = min(len(self.all_files), self.max_docs)
        return num_docs

    def _can_read(self, fpath):
        return fpath.lower().endswith(".csv")
