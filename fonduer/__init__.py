import logging

from fonduer._version import __version__
from fonduer.candidates import CandidateExtractor, OmniFigures, OmniNgrams
from fonduer.candidates.matchers import (
    DateMatcher,
    DictionaryMatch,
    Intersect,
    Inverse,
    LambdaFunctionFigureMatcher,
    LambdaFunctionMatcher,
    LocationMatcher,
    MiscMatcher,
    NumberMatcher,
    OrganizationMatcher,
    PersonMatcher,
    RegexMatchEach,
    RegexMatchSpan,
    Union,
)
from fonduer.candidates.models import candidate_subclass
from fonduer.learning import GenerativeModel, SparseLogisticRegression
from fonduer.meta import Meta
from fonduer.parser import Parser
from fonduer.parser.models import Document, Figure, Sentence, Table
from fonduer.parser.preprocessors import HTMLDocPreprocessor
from fonduer.supervision.annotations import load_gold_labels
from fonduer.supervision.async_annotations import (
    BatchFeatureAnnotator,
    BatchLabelAnnotator,
)

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "BatchFeatureAnnotator",
    "BatchLabelAnnotator",
    "CandidateExtractor",
    "DateMatcher",
    "DictionaryMatch",
    "Document",
    "Figure",
    "GenerativeModel",
    "HTMLDocPreprocessor",
    "Intersect",
    "Inverse",
    "LambdaFunctionFigureMatcher",
    "LambdaFunctionMatcher",
    "LocationMatcher",
    "Meta",
    "MiscMatcher",
    "NumberMatcher",
    "OmniFigures",
    "OmniNgrams",
    "OrganizationMatcher",
    "PDFPreprocessor",
    "Parser",
    "PersonMatcher",
    "RegexMatchEach",
    "RegexMatchSpan",
    "Sentence",
    "SparseLogisticRegression",
    "Table",
    "Union",
    "__version__",
    "candidate_subclass",
    "load_gold_labels",
]
