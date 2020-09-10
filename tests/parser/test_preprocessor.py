from fonduer.parser.preprocessors.hocr_doc_preprocessor import HOCRDocPreprocessor


def test_hocrpreprocessor():
    path = "tests/data/hocr_simple/md.hocr"
    preprocessor = HOCRDocPreprocessor(path=path)
    doc = next(preprocessor._parse_file(path, "md"))
    assert doc.name == "md"
