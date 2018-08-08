import logging

from fonduer._version import __version__
from fonduer.candidates import (
    CandidateExtractor,
    MentionExtractor,
    MentionFigures,
    MentionNgrams,
)
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
from fonduer.candidates.models import candidate_subclass, mention_subclass
from fonduer.learning import GenerativeModel, LogisticRegression
from fonduer.meta import Meta
from fonduer.parser import Parser
from fonduer.parser.models import Document, Figure, Sentence, Table
from fonduer.parser.preprocessors import HTMLDocPreprocessor
from fonduer.supervision.annotations import (
    FeatureAnnotator,
    LabelAnnotator,
    load_gold_labels,
)

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "CandidateExtractor",
    "DateMatcher",
    "DictionaryMatch",
    "Document",
    "FeatureAnnotator",
    "Figure",
    "GenerativeModel",
    "HTMLDocPreprocessor",
    "Intersect",
    "Inverse",
    "LabelAnnotator",
    "LambdaFunctionFigureMatcher",
    "LambdaFunctionMatcher",
    "LocationMatcher",
    "MentionExtractor",
    "MentionFigures",
    "MentionNgrams",
    "Meta",
    "MiscMatcher",
    "NumberMatcher",
    "OrganizationMatcher",
    "PDFPreprocessor",
    "Parser",
    "PersonMatcher",
    "RegexMatchEach",
    "RegexMatchSpan",
    "Sentence",
    "LogisticRegression",
    "Table",
    "Union",
    "__version__",
    "candidate_subclass",
    "load_gold_labels",
    "mention_subclass",
]
