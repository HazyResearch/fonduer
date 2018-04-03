import logging

from fonduer._version import __version__
# If we want to directly expose any subclasses at the package level, include
# them here. For example, this allows the user to write
# `from fonduer import HTMLPreprocessor` rather than having
# `from fonduer.parser import HTMLPreprocessor`
from fonduer.async_annotations import (BatchFeatureAnnotator,
                                       BatchLabelAnnotator, load_gold_labels)
from fonduer.candidates import CandidateExtractor, OmniFigures, OmniNgrams
from fonduer.learning import GenerativeModel, SparseLogisticRegression
from fonduer.matchers import (DateMatcher, DictionaryMatch, Intersect, Inverse,
                              LambdaFunctionFigureMatcher,
                              LambdaFunctionMatcher, LocationMatcher,
                              MiscMatcher, NumberMatcher, OrganizationMatcher,
                              PersonMatcher, RegexMatchEach, RegexMatchSpan,
                              Union)
from fonduer.models import Document, Figure, Meta, Phrase, candidate_subclass
from fonduer.parser import HTMLPreprocessor, OmniParser

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    'BatchFeatureAnnotator',
    'BatchLabelAnnotator',
    'CandidateExtractor',
    'DateMatcher',
    'DictionaryMatch',
    'Document',
    'Figure',
    'GenerativeModel',
    'HTMLPreprocessor',
    'Intersect',
    'Inverse',
    'LambdaFunctionFigureMatcher',
    'LambdaFunctionMatcher',
    'LocationMatcher',
    'Meta',
    'MiscMatcher',
    'NumberMatcher',
    'OmniFigures',
    'OmniNgrams',
    'OmniParser',
    'OrganizationMatcher',
    'PDFPreprocessor',
    'PersonMatcher',
    'Phrase',
    'RegexMatchEach',
    'RegexMatchSpan',
    'SparseLogisticRegression',
    'Union',
    '__version__',
    'candidate_subclass',
    'load_gold_labels',
]
