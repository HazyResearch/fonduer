"""Fonduer hOCR document preprocessor."""
import codecs
import os
import sys
from typing import Iterator, Optional, Tuple

from bs4 import BeautifulSoup
from bs4.element import Tag

from fonduer.parser.models import Document
from fonduer.parser.preprocessors.doc_preprocessor import DocPreprocessor


class HOCRDocPreprocessor(DocPreprocessor):
    """A ``Document`` generator for hOCR files.

    hOCR should comply with `hOCR v1.2`_.
    Note that *ppageno* property of *ocr_page* is optional by `hOCR v1.2`_,
    but is required by Fonduer.

    .. _hOCR v1.2: http://kba.cloud/hocr-spec/1.2/
    """

    def __init__(
        self,
        path: str,
        encoding: str = "utf-8",
        max_docs: int = sys.maxsize,
        space: bool = True,
    ):
        """Initialize HOCRDocPreprocessor.

        :param path: a path to file or directory, or a glob pattern. The basename
            (as returned by ``os.path.basename``) should be unique among all files.
        :param encoding: file encoding to use, defaults to "utf-8".
        :param max_docs: the maximum number of ``Documents`` to produce,
            defaults to sys.maxsize.
        :param space: boolean value indicating whether each word should have a
            subsequent space. E.g., English has spaces between words.
        :return: A generator of ``Documents``.
        """
        super().__init__(path, encoding, max_docs)
        self.space = space

    def _parse_file(self, fp: str, file_name: str) -> Iterator[Document]:
        # Adapted from https://github.com/ocropus/hocr-tools/blob/v1.3.0/hocr-check
        def get_prop(node: Tag, name: str) -> Optional[str]:
            title = node["title"]
            if not title:
                return None
            props = title.split(";")
            for prop in props:
                (key, args) = prop.split(None, 1)
                if key == name:
                    return args
            return None

        # Adapted from https://github.com/ocropus/hocr-tools/blob/v1.3.0/hocr-check
        def get_bbox(node: Tag) -> Tuple[str, ...]:
            bbox = get_prop(node, "bbox")
            if not bbox:
                return None
            return tuple([x for x in bbox.split()])

        with codecs.open(fp, encoding=self.encoding) as f:
            soup = BeautifulSoup(f, "lxml")
        all_html_elements = soup.find_all("html")
        if len(all_html_elements) != 1:
            raise NotImplementedError(
                f"Expecting exactly one html element per html file: {file_name}"
            )
        root = all_html_elements[0]
        capabilities = root.find("meta", attrs={"name": "ocr-capabilities"})
        if capabilities and "ocr_line" in capabilities["content"]:
            for line in root.find_all(class_="ocr_line"):
                line.unwrap()
        if capabilities and "ocrx_word" in capabilities["content"]:
            for p, page in enumerate(root.find_all(class_="ocr_page")):
                ppageno = str(p)  # 0-based
                for word in page.find_all(class_="ocrx_word"):
                    parent = word.parent
                    (left, top, right, bottom) = get_bbox(word)
                    if "left" not in parent.attrs:
                        parent["left"] = [left]
                        parent["top"] = [top]
                        parent["right"] = [right]
                        parent["bottom"] = [bottom]
                        parent["ppageno"] = [ppageno]
                        parent["tokens"] = [word.text]
                    else:
                        parent["left"].append(left)
                        parent["top"].append(top)
                        parent["right"].append(right)
                        parent["bottom"].append(bottom)
                        parent["ppageno"].append(ppageno)
                        parent["tokens"].append(word.text)
                    if "ocrp_wconf" in capabilities["content"]:
                        x_wconf = get_prop(word, "x_wconf")
                        if "x_wconf" not in parent.attrs:
                            parent["x_wconf"] = []
                        parent["x_wconf"].append(x_wconf)
                    # Mark the parent element
                    if "fonduer" not in parent.attrs:
                        parent["fonduer"] = ["1"]
                    word.unwrap()
            for parent in root.find_all(attrs={"fonduer": "1"}):
                if self.space:
                    parent.string = parent.text.replace("\n", " ").strip()
                else:
                    parent.string = parent.text.replace("\n", "").strip()
                # Rmove the mark
                del parent["fonduer"]
        name = os.path.basename(fp)[: os.path.basename(fp).rfind(".")]
        stable_id = self._get_stable_id(name)
        yield Document(
            name=name,
            stable_id=stable_id,
            text=str(root),
            meta={"file_name": file_name},
        )

    def __len__(self) -> int:
        """Provide a len attribute based on max_docs and number of files in folder."""
        num_docs = min(len(self.all_files), self.max_docs)
        return num_docs

    def _can_read(self, fpath: str) -> bool:
        """Return True if the path ends with either 'html' or 'hocr'."""
        return fpath.lower().endswith("hocr") or fpath.lower().endswith("html")
