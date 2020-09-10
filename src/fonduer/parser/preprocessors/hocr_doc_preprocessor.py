"""Fonduer HTML document preprocessor."""
import codecs
import os
from typing import Iterator

from bs4 import BeautifulSoup

from fonduer.parser.models import Document
from fonduer.parser.preprocessors.doc_preprocessor import DocPreprocessor


# Adapted from https://github.com/ocropus/hocr-tools/blob/v1.3.0/hocr-check#L29-L38
def get_prop(node, name):
    title = node["title"]
    if not title:
        return None
    props = title.split(";")
    for prop in props:
        (key, args) = prop.split(None, 1)
        if key == name:
            return args
    return None


# Adapted from https://github.com/ocropus/hocr-tools/blob/v1.3.0/hocr-check#L41-L45
def get_bbox(node):
    bbox = get_prop(node, "bbox")
    if not bbox:
        return None
    return tuple([x for x in bbox.split()])


class HOCRDocPreprocessor(DocPreprocessor):
    """A ``Document`` generator for hOCR files."""

    def _parse_file(self, fp: str, file_name: str) -> Iterator[Document]:
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
                for word in root.find_all(class_="ocrx_word"):
                    parent = word.parent
                    (left, top, right, bottom) = get_bbox(word)
                    if "left" in parent.attrs:
                        parent["left"] += " " + left
                        parent["top"] += " " + top
                        parent["right"] += " " + right
                        parent["bottom"] += " " + bottom
                    else:
                        parent["left"] = left
                        parent["top"] = top
                        parent["right"] = right
                        parent["bottom"] = bottom
                    if "ocrp_wconf" in capabilities["content"]:
                        x_wconf = get_prop(word, "x_wconf")
                        parent["x_wconf"] = (
                            parent["x_wconf"] + " " + x_wconf
                            if "x_wconf" in parent.attrs
                            else x_wconf
                        )
                    word.unwrap()
            name = os.path.basename(fp)[: os.path.basename(fp).rfind(".")]
            stable_id = self._get_stable_id(name)
            yield Document(
                name=name,
                stable_id=stable_id,
                text=root.prettify(),
                meta={"file_name": file_name},
            )

    def __len__(self) -> int:
        """Provide a len attribute based on max_docs and number of files in folder."""
        num_docs = min(len(self.all_files), self.max_docs)
        return num_docs

    def _can_read(self, fpath: str) -> bool:
        return fpath.lower().endswith("hocr")  # includes both .html and .xhtml
