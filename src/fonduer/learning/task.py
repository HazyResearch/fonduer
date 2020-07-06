"""Customized Emmental task for Fonduer."""
import logging
from functools import partial
from typing import Any, Dict, List, Optional, Union

from emmental.modules.embedding_module import EmbeddingModule
from emmental.modules.rnn_module import RNN
from emmental.modules.sparse_linear_module import SparseLinear
from emmental.scorer import Scorer
from emmental.task import EmmentalTask
from torch import Tensor, nn as nn
from torch.nn import functional as F

from fonduer.learning.modules.concat_linear import ConcatLinear
from fonduer.learning.modules.soft_cross_entropy_loss import SoftCrossEntropyLoss
from fonduer.utils.config import get_config

logger = logging.getLogger(__name__)


sce_loss = SoftCrossEntropyLoss()


def loss(
    module_name: str,
    intermediate_output_dict: Dict[str, Any],
    Y: Tensor,
    active: Tensor,
) -> Tensor:
    """Define the loss of the task.

    :param module_name: The module name to calculate the loss.
    :param intermediate_output_dict: The intermediate output dictionary
    :param Y: Ground truth labels.
    :param active: The sample mask.
    :return: Loss.
    """
    if len(Y.size()) == 1:
        label = intermediate_output_dict[module_name][0].new_zeros(
            intermediate_output_dict[module_name][0].size()
        )
        label.scatter_(1, Y.view(Y.size()[0], 1), 1.0)
    else:
        label = Y

    return sce_loss(intermediate_output_dict[module_name][0][active], label[active])


def output(module_name: str, intermediate_output_dict: Dict[str, Any]) -> Tensor:
    """Define the output of the task.

    :param module_name: The module name to calculate the loss.
    :param intermediate_output_dict: The intermediate output dictionary
    :return: Output tensor.
    """
    return F.softmax(intermediate_output_dict[module_name][0])


def create_task(
    task_names: Union[str, List[str]],
    n_arities: Union[int, List[int]],
    n_features: int,
    n_classes: Union[int, List[int]],
    emb_layer: Optional[EmbeddingModule],
    model: str = "LSTM",
    mode: str = "MTL",
) -> List[EmmentalTask]:
    """Create task from relation(s).

    :param task_names: Relation name(s), If str, only one relation; If List[str],
        multiple relations.
    :param n_arities: The arity of each relation.
    :param n_features: The multimodal feature set size.
    :param n_classes: Number of classes for each task. (Only support classification
        task now).
    :param emb_layer: The embedding layer for LSTM. No need for LogisticRegression
        model.
    :param model: Model name (available models: "LSTM", "LogisticRegression"),
        defaults to "LSTM".
    :param mode: Learning mode (available modes: "STL", "MTL"),
        defaults to "MTL".
    """
    if model not in ["LSTM", "LogisticRegression"]:
        raise ValueError(
            f"Unrecognized model {model}. Only support {['LSTM', 'LogisticRegression']}"
        )

    if mode not in ["STL", "MTL"]:
        raise ValueError(f"Unrecognized mode {mode}. Only support {['STL', 'MTL']}")

    config = get_config()["learning"][model]
    logger.info(f"{model} model config: {config}")

    if not isinstance(task_names, list):
        task_names = [task_names]
    if not isinstance(n_arities, list):
        n_arities = [n_arities]
    if not isinstance(n_classes, list):
        n_classes = [n_classes]

    tasks = []

    for task_name, n_arity, n_class in zip(task_names, n_arities, n_classes):
        if mode == "MTL":
            feature_module_name = "shared_feature"
        else:
            feature_module_name = f"{task_name}_feature"

        if model == "LSTM":
            module_pool = nn.ModuleDict(
                {
                    "emb": emb_layer,
                    feature_module_name: SparseLinear(
                        n_features + 1, config["hidden_dim"], bias=config["bias"]
                    ),
                }
            )
            for i in range(n_arity):
                module_pool.update(
                    {
                        f"{task_name}_lstm{i}": RNN(
                            num_classes=0,
                            emb_size=emb_layer.dim,
                            lstm_hidden=config["hidden_dim"],
                            attention=config["attention"],
                            dropout=config["dropout"],
                            bidirectional=config["bidirectional"],
                        )
                    }
                )
            module_pool.update(
                {
                    f"{task_name}_pred_head": ConcatLinear(
                        [f"{task_name}_lstm{i}" for i in range(n_arity)]
                        + [feature_module_name],
                        config["hidden_dim"] * (2 * n_arity + 1)
                        if config["bidirectional"]
                        else config["hidden_dim"] * (n_arity + 1),
                        n_class,
                    )
                }
            )

            task_flow = []
            task_flow += [
                {
                    "name": f"{task_name}_emb{i}",
                    "module": "emb",
                    "inputs": [("_input_", f"m{i}")],
                }
                for i in range(n_arity)
            ]
            task_flow += [
                {
                    "name": f"{task_name}_lstm{i}",
                    "module": f"{task_name}_lstm{i}",
                    "inputs": [(f"{task_name}_emb{i}", 0), ("_input_", f"m{i}_mask")],
                }
                for i in range(n_arity)
            ]
            task_flow += [
                {
                    "name": feature_module_name,
                    "module": feature_module_name,
                    "inputs": [
                        ("_input_", "feature_index"),
                        ("_input_", "feature_weight"),
                    ],
                }
            ]
            task_flow += [
                {
                    "name": f"{task_name}_pred_head",
                    "module": f"{task_name}_pred_head",
                    "inputs": None,
                }
            ]
        elif model == "LogisticRegression":
            module_pool = nn.ModuleDict(
                {
                    feature_module_name: SparseLinear(
                        n_features + 1, config["hidden_dim"], bias=config["bias"]
                    ),
                    f"{task_name}_pred_head": ConcatLinear(
                        [feature_module_name], config["hidden_dim"], n_class
                    ),
                }
            )

            task_flow = [
                {
                    "name": feature_module_name,
                    "module": feature_module_name,
                    "inputs": [
                        ("_input_", "feature_index"),
                        ("_input_", "feature_weight"),
                    ],
                },
                {
                    "name": f"{task_name}_pred_head",
                    "module": f"{task_name}_pred_head",
                    "inputs": None,
                },
            ]
        else:
            raise ValueError(f"Unrecognized model {model}.")

        tasks.append(
            EmmentalTask(
                name=task_name,
                module_pool=module_pool,
                task_flow=task_flow,
                loss_func=partial(loss, f"{task_name}_pred_head"),
                output_func=partial(output, f"{task_name}_pred_head"),
                scorer=Scorer(metrics=["accuracy", "precision", "recall", "f1"]),
            )
        )

    return tasks
