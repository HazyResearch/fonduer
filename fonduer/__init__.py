from fonduer._version import __version__

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# If we want to directly expose any subclasses at the package level, include
# them here. For example, this allows the user to write
# `from fonduer import HTMLPreprocessor` rather than having
# `from fonduer.parser import HTMLPreprocessor`
from fonduer.async_annotations import BatchFeatureAnnotator
from fonduer.async_annotations import BatchLabelAnnotator
from fonduer.candidates import CandidateExtractor, OmniNgrams, OmniFigures
from fonduer.matchers import LambdaFunctionFigureMatcher
from fonduer.models import (SnorkelSession, Document, Phrase, Figure,
                            candidate_subclass)
from fonduer.parser import HTMLPreprocessor
from fonduer.parser import OmniParser
from fonduer.snorkel.annotations import load_gold_labels
from fonduer.snorkel.learning import GenerativeModel, SparseLogisticRegression
from fonduer.snorkel.matchers import (RegexMatchSpan, DictionaryMatch,
                                      LambdaFunctionMatcher, Intersect, Union)

# Raise the visibility of these subpackages to the package level for cleaner
# syntax. The key idea here is when we do `from package.submodule1 import foo`
# `sys.modules` is checked to see if it has package. If not, then
# `package/__init__.py` is run and loaded. Then, the process repeats for
# `package.submodule1`. We can omit the fonduer submodule in the path by
# using this __init__.py to put these packages in sys.modules directly.
#
# This allows `from fonduer.models import Phrase` rather than
# need to write `from fonduer.fonduer.models import Phrase`
#  from .fonduer import models
#  from .fonduer import lf_helpers
#  from .fonduer import visualizer
#  from .fonduer import candidates
#
#  for module in [models, lf_helpers, visualizer, candidates]:
#      full_name = '{}.{}'.format(__package__, module.__name__.rsplit('.')[-1])
#      sys.modules[full_name] = sys.modules[module.__name__]
