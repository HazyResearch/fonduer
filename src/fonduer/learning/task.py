import logging
from functools import partial
from typing import Any, Dict, List, Optional, Union

import torch.nn as nn
import torch.nn.functional as F
from emmental.modules.embedding_module import EmbeddingModule
from emmental.modules.rnn_module import RNN
from emmental.modules.sparse_linear_module import SparseLinear
from emmental.scorer import Scorer
from emmental.task import EmmentalTask
from torch import Tensor

from fonduer.learning.modules.soft_cross_entropy_loss import SoftCrossEntropyLoss
from fonduer.learning.modules.sum_module import Sum_module
from fonduer.utils.config import get_config

logger = logging.getLogger(__name__)


sce_loss = SoftCrossEntropyLoss()


def loss(
    module_name: str,
    intermediate_output_dict: Dict[str, Any],
    Y: Tensor,
    active: Tensor,
) -> Tensor:
    if len(Y.size()) == 1:
        label = intermediate_output_dict[module_name][0].new_zeros(
            intermediate_output_dict[module_name][0].size()
        )
        label.scatter_(1, (Y - 1).view(Y.size()[0], 1), 1.0)
    else:
        label = Y

    return sce_loss(intermediate_output_dict[module_name][0][active], label[active])


def output(module_name: str, intermediate_output_dict: Dict[str, Any]) -> Tensor:
    return F.softmax(intermediate_output_dict[module_name][0])


def create_task(
    task_names: Union[str, List[str]],
    n_arities: Union[int, List[int]],
    n_features: int,
    n_classes: Union[int, List[int]],
    emb_layer: Optional[EmbeddingModule],
    model: str = "LSTM",
) -> List[EmmentalTask]:
    """Create task from relation(s).

    :param task_names: Relation name(s), If str, only one relation; If List[str],
        multiple relations.
    :type task_names: str, List[str]
    :param n_arities: The arity of each relation.
    :type n_arities: int, List[int]
    :param n_features: The multimodal feature set size.
    :type n_features: int
    :param n_classes: Number of classes for each task. (Only support classification
        task now).
    :type n_classes: int, List[int]
    :param emb_layer: The embedding layer for LSTM. No need for LogisticRegression
        model.
    :type emb_layer: EmbeddingModule
    :param model: Model name (available models: "LSTM", "LogisticRegression"),
        defaults to "LSTM".
    :type model: str

    """

    if model not in ["LSTM", "LogisticRegression"]:
        raise ValueError(
            f"Unrecognized model {model}. Only support {['LSTM', 'LogisticRegression']}"
        )

    config = get_config()["learning"][model]
    logger.info(f"{model} model config: {config}")

    if not isinstance(task_names, list):
        task_names = [task_names]
        n_arities = [n_arities]
        n_classes = [n_classes]

    tasks = []

    for task_name, n_arity, n_class in zip(task_names, n_arities, n_classes):
        if model == "LSTM":
            module_pool = nn.ModuleDict(
                {
                    "emb": emb_layer,
                    "feature": SparseLinear(
                        n_features + 1, n_class, bias=config["bias"]
                    ),
                }
            )
            for i in range(n_arity):
                module_pool.update(
                    {
                        f"lstm{i}": RNN(
                            num_classes=n_class,
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
                    f"{task_name}_pred_head": Sum_module(
                        [f"lstm{i}" for i in range(n_arity)] + ["feature"]
                    )
                }
            )

            task_flow = []  # type:ignore
            task_flow += [
                {"name": f"emb{i}", "module": "emb", "inputs": [("_input_", f"m{i}")]}
                for i in range(n_arity)
            ]
            task_flow += [
                {
                    "name": f"lstm{i}",
                    "module": f"lstm{i}",
                    "inputs": [(f"emb{i}", 0), ("_input_", f"m{i}_mask")],
                }
                for i in range(n_arity)
            ]
            task_flow += [
                {
                    "name": "feature",
                    "module": "feature",
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
                    "feature": SparseLinear(
                        n_features + 1, n_class, bias=config["bias"]
                    ),
                    f"{task_name}_pred_head": Sum_module(["feature"]),
                }
            )

            task_flow = [
                {
                    "name": "feature",
                    "module": "feature",
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
