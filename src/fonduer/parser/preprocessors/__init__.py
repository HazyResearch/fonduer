"""Fonduer's parser preprocessor module."""
from fonduer.parser.preprocessors.csv_doc_preprocessor import CSVDocPreprocessor
from fonduer.parser.preprocessors.doc_preprocessor import DocPreprocessor
from fonduer.parser.preprocessors.hocr_doc_preprocessor import HOCRDocPreprocessor
from fonduer.parser.preprocessors.html_doc_preprocessor import HTMLDocPreprocessor
from fonduer.parser.preprocessors.text_doc_preprocessor import TextDocPreprocessor
from fonduer.parser.preprocessors.tsv_doc_preprocessor import TSVDocPreprocessor

__all__ = [
    "CSVDocPreprocessor",
    "DocPreprocessor",
    "HOCRDocPreprocessor",
    "HTMLDocPreprocessor",
    "TSVDocPreprocessor",
    "TextDocPreprocessor",
]
