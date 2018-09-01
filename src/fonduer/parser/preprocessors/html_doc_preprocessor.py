import codecs
import os

from bs4 import BeautifulSoup

from fonduer.parser.models import Document
from fonduer.parser.preprocessors.doc_preprocessor import DocPreprocessor


class HTMLDocPreprocessor(DocPreprocessor):
    """Simple parsing of files into html documents"""

    def __init__(self, *args, **kwargs):
        matching_pdf_folder = kwargs.pop("matching_pdf_folder", None)
        super(HTMLDocPreprocessor, self).__init__(*args, **kwargs)
        if matching_pdf_folder:
            pdf_files = [
                x.lower()
                for x in os.listdir(matching_pdf_folder)
                if x.lower().endswith(".pdf")
            ]
            for file in self.all_files:
                pdf_file_name = os.path.basename(file).replace("html", "pdf")
                if not pdf_file_name.lower() in pdf_files:
                    print(pdf_files)
                    raise FileNotFoundError(
                        "Could not find matching pdf file ({})for html file {}.".format(
                            pdf_file_name, file
                        )
                    )

    def parse_file(self, fp, file_name):
        with codecs.open(fp, encoding=self.encoding) as f:
            soup = BeautifulSoup(f, "lxml")
            for text in soup.find_all("html"):
                name = os.path.basename(fp)[: os.path.basename(fp).rfind(".")]
                stable_id = self.get_stable_id(name)
                yield Document(
                    name=name,
                    stable_id=stable_id,
                    text=str(text),
                    meta={"file_name": file_name},
                ), str(text)

    def _can_read(self, fpath):
        return fpath.endswith("html")  # includes both .html and .xhtml
