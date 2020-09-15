"""Unit tests for preprocessors."""

from fonduer.parser.preprocessors.hocr_doc_preprocessor import HOCRDocPreprocessor


def test_hocrpreprocessor():
    """Test hOCRDocPreprocessor."""
    path = "tests/data/hocr_simple/md.hocr"
    preprocessor = HOCRDocPreprocessor(path=path)
    doc = next(preprocessor.__iter__())
    assert doc.name == "md"
