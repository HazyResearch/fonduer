"""Unit tests for preprocessors."""
from bs4 import BeautifulSoup

from fonduer.parser.preprocessors.hocr_doc_preprocessor import HOCRDocPreprocessor


def test_hocrpreprocessor():
    """Test hOCRDocPreprocessor with a simple hOCR."""
    path = "tests/data/hocr_simple/md.hocr"
    preprocessor = HOCRDocPreprocessor(path=path)
    doc = next(iter(preprocessor))
    assert doc.name == "md"
    # the intermidiate attribute: "fonduer" should be removed.
    assert "fonduer" not in doc.text
    # number of "left" attribute is equal to that of "ppageno" - 1 (at ocr_page)
    assert doc.text.count("left") == doc.text.count("ppageno") - 1 == 33


def test_hocrpreprocessor_space_false():
    """Test hOCRDocPreprocessor with space=False."""
    path = "tests/data/hocr_simple/japan.hocr"
    preprocessor = HOCRDocPreprocessor(path=path, space=False)
    doc = next(iter(preprocessor))
    assert doc.name == "japan"
    # the intermidiate attribute: "fonduer" should be removed.
    assert "fonduer" not in doc.text

    soup = BeautifulSoup(doc.text, "lxml")
    element = soup.find(id="par_1_1")

    # A token cannot contain " " (whitespace) as "tokens" are deliminated by a " ".
    assert len(element.get("left").split()) == len(element.get("tokens").split()) == 59
