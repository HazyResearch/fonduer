"""Fonduer dataset."""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from emmental.data import EmmentalDataset
from scipy.sparse import csr_matrix
from torch import Tensor

from fonduer.candidates.models import Candidate
from fonduer.learning.utils import mark_sentence, mention_to_tokens

logger = logging.getLogger(__name__)


class FonduerDataset(EmmentalDataset):
    """A FonduerDataset class which is inherited from EmmentalDataset.

    This class takes list of candidates and corresponding feature matrix as input and
    wraps them.

    :param name: The name of the dataset.
    :param candidates: The list of candidates.
    :param features: The corresponding feature matrix.
    :param word2id: The name of the dataset.
    :param labels: If np.array, it's the label for all candidates; If int, it's
        the number of classes of label and we will create placeholder labels
        (mainly used for inference).
    :param labels: Which candidates to use. If None, use all candidates.
    """

    def __init__(
        self,
        name: str,
        candidates: List[Candidate],
        features: csr_matrix,
        word2id: Dict,
        labels: Union[np.array, int],
        index: Optional[List[int]] = None,
    ):
        """Initialize FonduerDataset."""
        self.name = name
        self.candidates = candidates
        self.features = features
        self.word2id = word2id
        self.labels = labels
        self.index = index

        self.X_dict: Dict[str, List[Any]] = {}
        self.Y_dict: Dict[str, Tensor] = {}

        self._map_to_id()
        self._map_features()
        self._map_labels()

        uids = [f"{self.name}_{idx}" for idx in range(len(self.candidates))]
        self.add_features({"_uids_": uids})

        super().__init__(name, self.X_dict, self.Y_dict, "_uids_")

    def __len__(self) -> int:
        """Get the length of the dataset."""
        try:
            if self.index is not None:
                return len(self.index)
            else:
                return len(next(iter(self.X_dict.values())))
        except StopIteration:
            return 0

    def __getitem__(
        self, index: int
    ) -> Tuple[Dict[str, Union[Tensor, list]], Dict[str, Tensor]]:
        """Get the data from dataset."""
        if self.index is not None:
            index = self.index[index]

        x_dict = {name: feature[index] for name, feature in self.X_dict.items()}
        y_dict = {name: label[index] for name, label in self.Y_dict.items()}
        return x_dict, y_dict

    def _map_to_id(self) -> None:
        self.X_dict.update(
            dict([(f"m{i}", []) for i in range(len(self.candidates[0]))])
        )

        for candidate in self.candidates:
            for i in range(len(candidate)):
                # Add mark for each mention in the original sentence
                args = [
                    (
                        candidate[i].context.get_word_start_index(),
                        candidate[i].context.get_word_end_index(),
                        i,
                    )
                ]
                s = mark_sentence(mention_to_tokens(candidate[i]), args)
                self.X_dict[f"m{i}"].append(
                    torch.tensor(
                        [
                            self.word2id[w]
                            if w in self.word2id
                            else self.word2id["<unk>"]
                            for w in s
                        ],
                        dtype=torch.long,
                    )
                )

    def _map_features(self) -> None:
        self.X_dict.update({"feature_index": [], "feature_weight": []})
        for i in range(len(self.candidates)):
            self.X_dict["feature_index"].append(
                torch.tensor(
                    self.features.indices[
                        self.features.indptr[i] : self.features.indptr[i + 1]
                    ],
                    dtype=torch.long,
                )
                + 1
            )
            self.X_dict["feature_weight"].append(
                torch.tensor(
                    self.features.data[
                        self.features.indptr[i] : self.features.indptr[i + 1]
                    ],
                    dtype=torch.float,
                )
            )

    def _map_labels(self) -> None:
        if isinstance(self.labels, int):
            self.Y_dict.update(
                {
                    "labels": torch.from_numpy(
                        np.random.randint(self.labels, size=len(self.candidates))
                    )
                }
            )
        else:
            self.Y_dict.update({"labels": torch.tensor(np.array(self.labels))})
