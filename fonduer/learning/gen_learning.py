from metal import analysis
from metal.label_model import LabelModel
from metal.multitask import MTLabelModel, TaskHierarchy


def GenerativeModel(cardinalities=2, dependencies=[], **kwargs):
    """
    A generative factory function for data programming (MeTal).
    """
    # Use single task label model if cardinality is an integer,
    # Otherwise use multi task label model
    if isinstance(cardinalities, int):
        label_model = LabelModel(k=cardinalities, **kwargs)
    else:
        task_graph = TaskHierarchy(cardinalities=cardinalities, edges=dependencies)
        label_model = MTLabelModel(task_graph=task_graph, **kwargs)

    return label_model


def GenerativeModelAnalyzer():
    """
    A analysis factory function for MeTaL.
    """
    return analysis
