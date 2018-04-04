"""
Subpackage for Snorkel machine learning modules.
"""
from fonduer.learning.disc_models.logistic_regression import (LogisticRegression,
                                                              SparseLogisticRegression)
from fonduer.learning.disc_models.rnn import TextRNN, reRNN
from fonduer.learning.gen_learning import (GenerativeModel,
                                           GenerativeModelWeights)
from fonduer.learning.utils import (GridSearch, LabelBalancer, LF_accuracies,
                                    LF_conflicts, LF_coverage, LF_overlaps,
                                    MentionScorer, ModelTester, RandomSearch,
                                    Scorer, binary_scores_from_counts,
                                    candidate_conflict, candidate_coverage,
                                    candidate_overlap, print_scores,
                                    reshape_marginals, sparse_abs,
                                    training_set_summary_stats)

__all__ = [
    'GenerativeModel',
    'GenerativeModelWeights',
    'GridSearch',
    'LF_accuracies',
    'LF_conflicts',
    'LF_coverage',
    'LF_overlaps',
    'LabelBalancer',
    'LogisticRegression',
    'MentionScorer',
    'ModelTester',
    'RandomSearch',
    'Scorer',
    'SparseLogisticRegression',
    'TextRNN',
    'binary_scores_from_counts',
    'candidate_conflict',
    'candidate_converage',
    'candidate_coverage',
    'candidate_overlap',
    'print_scores',
    'reRNN',
    'reshape_marginals',
    'sparse_abs',
    'training_set_summary_stats',
]
