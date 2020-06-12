"""Fonduer CSV document preprocessor."""
import codecs
import csv
import os
import sys
from typing import Callable, Dict, Iterator, Optional

from fonduer.parser.models import Document
from fonduer.parser.preprocessors.doc_preprocessor import DocPreprocessor
from fonduer.utils.utils_parser import build_node, column_constructor


class CSVDocPreprocessor(DocPreprocessor):
    """A ``Document`` generator for CVS files.

    It treats each line in the input file as a ``Document``.
    It assumes that each column is one ``Section`` and content in each column as one
    ``Paragraph`` by default. However, if the column is complex, an advanced parser
    may be used by specifying ``parser_rule`` parameter in a dict format where key
    is the column index and value is the specific parser, e,g., ``column_constructor``
    in ``fonduer.utils.utils_parser``.
    """

    def __init__(
        self,
        path: str,
        encoding: str = "utf-8",
        max_docs: int = sys.maxsize,
        header: bool = False,
        delim: str = ",",
        parser_rule: Optional[Dict[int, Callable]] = None,
    ) -> None:
        """Initialize CSV DocPreprocessor.

        :param path: a path to file or directory, or a glob pattern. The basename
            (as returned by ``os.path.basename``) should be unique among all files.
        :param encoding: file encoding to use (e.g. "utf-8").
        :param max_docs: the maximum number of ``Documents`` to produce.
        :param header: if the CSV file contain header or not, if yes, the header
            will be used as Section name. default = False
        :param delim: delimiter to be used to separate columns when file has
            more than one column. It is active only when ``column is not
            None``. default=','
        :param parser_rule: The parser rule to be used to parse the specific column.
            default = None
        :return: A generator of ``Documents``.
        """
        super().__init__(path, encoding, max_docs)
        self.header = header
        self.delim = delim
        self.parser_rule = parser_rule

    def _parse_file(self, fp: str, file_name: str) -> Iterator[Document]:
        name = os.path.basename(fp)[: os.path.basename(fp).rfind(".")]
        with codecs.open(fp, encoding=self.encoding) as f:
            reader = csv.reader(f)

            # Load CSV header
            header_names = None
            if self.header:
                header_names = next(reader)

            # Load document per row
            for i, row in enumerate(reader):
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
                    context = [
                        build_node(t, n, c)
                        # TODO: Fix this type ignore
                        for t, n, c in rule(content)  # type: ignore
                    ]
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

    def __len__(self) -> int:
        """Provide a len attribute based on max_docs and number of rows in files."""
        cnt_docs = 0
        for fp in self.all_files:
            with codecs.open(fp, encoding=self.encoding) as csv:
                num_lines = sum(1 for line in csv)
                cnt_docs += num_lines - 1 if self.header else num_lines
            if cnt_docs > self.max_docs:
                break
        num_docs = min(cnt_docs, self.max_docs)
        return num_docs

    def _can_read(self, fpath: str) -> bool:
        return fpath.lower().endswith(".csv")
