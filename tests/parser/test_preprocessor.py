"""Unit tests for preprocessors."""

from fonduer.parser.preprocessors.hocr_doc_preprocessor import HOCRDocPreprocessor


def test_hocrpreprocessor():
    """Test hOCRDocPreprocessor."""
    path = "tests/data/hocr_simple/md.hocr"
    preprocessor = HOCRDocPreprocessor(path=path)
    doc = next(iter(preprocessor))
    assert doc.name == "md"


def test_hocrpreprocessor_wo_ppageno():
    """Test hOCRDocPreprocessor."""
    path = "tests/data/hocr_simple/japan.hocr"
    preprocessor = HOCRDocPreprocessor(path=path, space=False)
    doc = next(iter(preprocessor))
    assert doc.name == "japan"
