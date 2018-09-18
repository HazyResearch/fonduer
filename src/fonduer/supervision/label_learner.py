import logging

from metal import analysis
from metal.label_model import LabelModel
from metal.multitask import MTLabelModel, TaskHierarchy

logger = logging.getLogger(__name__)


def LabelLearner(cardinalities=2, dependencies=[], **kwargs):
    """
    A generative factory function for data programming (MeTal).

    Specifically, this is a wrapper for the LabelModel and MTLabelModel in
    ``metal``. See https://github.com/HazyResearch/metal/.

    :param cardinalities: In the single task case, a single integer for the
        number of classes in the task. In the multi-task case, a t-length list
        of integers corresponding to the classes of each task.
    :type cardinalities: int or list of int
    :param dependencies: A list of (a,b) tuples meaning a is a parent of b in a
        tree.
    :type dependencies: list
    """
    # Use single task label model if cardinality is an integer,
    # Otherwise use multi task label model
    if isinstance(cardinalities, int):
        logger.info("Using MeTaL single task label model")
        label_model = LabelModel(k=cardinalities, **kwargs)
    else:
        logger.info("Using MeTaL multi task label model")
        task_graph = TaskHierarchy(cardinalities=cardinalities, edges=dependencies)
        label_model = MTLabelModel(task_graph=task_graph, **kwargs)

    return label_model


def LabelAnalyzer():
    """
    A analysis factory function for MeTaL.

    Specifically, a wrapper for ``metal.analysis``.
    See https://github.com/HazyResearch/metal/.
    """
    return analysis
