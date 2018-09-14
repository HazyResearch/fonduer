import codecs

from fonduer.parser.preprocessors.doc_preprocessor import DocPreprocessor
from fonduer.parser.preprocessors.text_doc_preprocessor import TextDocPreprocessor


class CSVPathsPreprocessor(DocPreprocessor):
    """This ``DocPreprocessor`` treats inputs file as index of paths to
     actual documents; each line in the input file contains a path to a document.

     **Defaults and Customization:**

     * The input file is treated as a simple text file having one path per file.
       However, if the input is a CSV file, a pair of ``column`` and ``delim``
       parameters may be used to retrieve the desired value as reference path.

     * The referenced documents are treated as text document and hence parsed
       using ``TextDocPreprocessor``. However, if the referenced files are
       complex, an advanced parser may be used by specifying ``parser_factory``
       parameter to constructor.

    :param path: input file having paths
    :param parser_factory: The parser class to be used to parse the
        referenced files. default = TextDocPreprocessor
    :param column: index of the column which references path. default=None,
        which implies that each line has only one column
    :param delim: delimiter to be used to separate columns when file has
        more than one column. It is active only when ``column is not
        None``. default=','

    :rtype: A generator of ``Documents``.
     """

    def __init__(
        self,
        path,
        parser_factory=TextDocPreprocessor,
        column=None,
        delim=",",
        *args,
        **kwargs
    ):
        super(CSVPathsPreprocessor, self).__init__(path, *args, **kwargs)
        self.column = column
        self.delim = delim
        self.parser = parser_factory(path)

    def _get_files(self, path):
        with codecs.open(path, encoding=self.encoding) as lines:
            for doc_path in lines:
                if self.column is not None:
                    # if column is set, retrieve specific column from CSV record
                    doc_path = doc_path.split(self.delim)[self.column]
                yield doc_path.strip()

    def _parse_file(self, fp, file_name):
        return self.parser.parse_file(fp, file_name)
