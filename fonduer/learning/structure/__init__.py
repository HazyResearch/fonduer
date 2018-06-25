"""
Subpackage for learning the structures of models.
"""
from fonduer.learning.gen_learning import DependencySelector
from fonduer.learning.structure.synthetic import generate_model, generate_label_matrix
from fonduer.learning.utils import get_deps, get_all_deps

__all__ = [
    "DependencySelector",
    "generate_model",
    "generate_label_matrix",
    "get_deps",
    "get_all_deps",
]
