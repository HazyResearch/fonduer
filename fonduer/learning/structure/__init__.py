"""
Subpackage for learning the structures of models.
"""
from __future__ import absolute_import

from .gen_learning import DependencySelector
from .synthetic import generate_model, generate_label_matrix
from .utils import get_deps, get_all_deps

__all__ = [
    'DependencySelector', 'generate_model', 'generate_label_matrix',
    'get_deps', 'get_all_deps'
]
