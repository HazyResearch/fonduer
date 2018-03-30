from fonduer._version import __version__

import logging

# If we want to directly expose any subclasses at the package level, include
# them here. For example, this allows the user to write
# `from fonduer import HTMLPreprocessor` rather than having
# `from fonduer.parser import HTMLPreprocessor`
from fonduer.async_annotations import BatchFeatureAnnotator
from fonduer.async_annotations import BatchLabelAnnotator
from fonduer.candidates import CandidateExtractor, OmniNgrams, OmniFigures
from fonduer.matchers import LambdaFunctionFigureMatcher
from fonduer.models import (Meta, Document, Phrase, Figure, candidate_subclass)
from fonduer.parser import HTMLPreprocessor
from fonduer.parser import OmniParser
from fonduer.snorkel.annotations import load_gold_labels
from fonduer.snorkel.learning import GenerativeModel, SparseLogisticRegression
from fonduer.snorkel.matchers import (RegexMatchSpan, DictionaryMatch,
                                      LambdaFunctionMatcher, Intersect, Union)

logging.getLogger(__name__).addHandler(logging.NullHandler())
__all__ = [
    '__version__', 'BatchFeatureAnnotator', 'BatchLabelAnnotator',
    'CandidateExtractor', 'OmniNgrams', 'OmniFigures',
    'LambdaFunctionFigureMatcher', 'Meta', 'Document', 'Phrase', 'Figure',
    'candidate_subclass', 'HTMLPreprocessor', 'OmniParser', 'load_gold_labels',
    'GenerativeModel', 'SparseLogisticRegression', 'RegexMatchSpan',
    'DictionaryMatch', 'LambdaFunctionMatcher', 'Intersect', 'Union'
]
