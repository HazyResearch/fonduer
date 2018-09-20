import glob
import os


class DocPreprocessor(object):
    """
    A generator which processes a file or directory of files into a set of
    Document objects.

    :param encoding: file encoding to use (e.g. "utf-8").
    :type encoding: str
    :param path: filesystem path to file or directory to parse.
    :type path: str
    :param max_docs: the maximum number of ``Documents`` to produce.
    :type max_docs: int
    :rtype: A generator of ``Documents``.
    """

    def __init__(self, path, encoding="utf-8", max_docs=float("inf")):
        self.path = path
        self.encoding = encoding
        self.max_docs = max_docs
        self.all_files = self._get_files(self.path)

    def _generate(self):
        """Parses a file or directory of files into a set of ``Document`` objects."""
        doc_count = 0
        for fp in self.all_files:
            for doc in self._get_docs_for_path(fp):
                yield doc
                doc_count += 1
                if doc_count >= self.max_docs:
                    return

    def __len__(self):
        raise NotImplementedError(
            "One generic file can yield more than one Document object, "
            "so length can not be yielded before we process all files"
        )

    def __iter__(self):
        return self._generate()

    def _get_docs_for_path(self, fp):
        file_name = os.path.basename(fp)
        if self._can_read(file_name):
            for doc in self._parse_file(fp, file_name):
                yield doc

    def _get_stable_id(self, doc_id):
        return "%s::document:0:0" % doc_id

    def _parse_file(self, fp, file_name):
        raise NotImplementedError()

    def _can_read(self, fpath):
        return not fpath.startswith(".")

    def _get_files(self, path):
        if os.path.isfile(path):
            fpaths = [path]
        elif os.path.isdir(path):
            fpaths = [os.path.join(path, f) for f in os.listdir(path)]
        else:
            fpaths = glob.glob(path)
        fpaths = [x for x in fpaths if self._can_read(x)]
        if len(fpaths) > 0:
            return sorted(fpaths)
        else:
            raise IOError("File or directory not found: {}".format(path))
