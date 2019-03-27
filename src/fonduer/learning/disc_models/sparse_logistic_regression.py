import numpy as np
import torch

from fonduer.learning.classifier import Classifier
from fonduer.learning.disc_models.modules.sparse_linear import SparseLinear
from fonduer.learning.disc_models.utils import pad_batch
from fonduer.utils.config import get_config


class SparseLogisticRegression(Classifier):
    """
    Sparse Logistic Regression model.

    :param name: User-defined name of the model.
    :type name: str
    """

    def forward(self, X):
        """
        Run forward pass.

        :param X: The input (batch) of the model contains features and feature weights.
        :type X: For features: torch.Tensor of shape (batch_size, sparse_feature_size).
            For feature weights: torch.Tensor of shape
            (batch_size, sparse_feature_size).
        :return: The output of sparse Logistic Regression layer.
        :rtype: torch.Tensor of shape (batch_size, num_classes)
        """

        return self.sparse_linear(X[0], X[1])

    def _check_input(self, X):
        """
        Check input format.

        :param X: The input data of the model.
        :type X: (candidates, features) pair
        :return: True if valid, otherwise False.
        :rtype: bool
        """

        return isinstance(X, tuple)

    def _preprocess_data(self, X, Y=None, idxs=None, train=False):
        """
        Preprocess the data:
        1. Select subset of the input if idxs exists.

        :param X: The input data of the model.
        :type X: pair with candidates and corresponding features
        :param Y: The labels of input data.
        :type Y: list or numpy.array
        :param idxs: The selected indices of input data.
        :type idxs: list or numpy.array
        :param train: Indicator of training set.
        :type train: bool
        :return: Preprocessed data.
        :rtype: list of (features, feature_weights) pair
        """

        C, F = X

        if Y is not None:
            Y = np.array(Y).astype(np.float32)

        if idxs is None:
            if Y is not None:
                return (
                    [
                        [
                            F.indices[F.indptr[i] : F.indptr[i + 1]],
                            F.data[F.indptr[i] : F.indptr[i + 1]],
                        ]
                        for i in range(len(C))
                    ],
                    Y,
                )
            else:
                return [
                    [
                        F.indices[F.indptr[i] : F.indptr[i + 1]],
                        F.data[F.indptr[i] : F.indptr[i + 1]],
                    ]
                    for i in range(len(C))
                ]
        if Y is not None:
            return (
                [
                    [
                        F.indices[F.indptr[i] : F.indptr[i + 1]],
                        F.data[F.indptr[i] : F.indptr[i + 1]],
                    ]
                    for i in idxs
                ],
                Y[idxs],
            )
        else:
            return [
                [
                    F.indices[F.indptr[i] : F.indptr[i + 1]],
                    F.data[F.indptr[i] : F.indptr[i + 1]],
                ]
                for i in idxs
            ]

    def _collate(self, batch):
        """
        Puts each data field into a tensor.

        :param batch: The input data batch.
        :type batch: list of (features, feature_weights) pair
        :return: Preprocessed data.
        :rtype: list of torch.Tensor with torch.Tensor (Optional)
        """

        Y_batch = None
        if isinstance(batch[0], tuple):
            batch, Y_batch = list(zip(*batch))
            Y_batch = self._cuda(torch.Tensor(Y_batch))

        f_batch, v_batch = list(zip(*batch))

        f_batch, _ = pad_batch(f_batch, 0)
        v_batch, _ = pad_batch(v_batch, 0, type="float")

        f_batch = self._cuda(f_batch)
        v_batch = self._cuda(v_batch)

        if Y_batch is not None:
            return [f_batch, v_batch], Y_batch
        else:
            return [f_batch, v_batch]

    def _update_settings(self, X):
        """
        Update the model argument.

        :param X: The input data of the model.
        :type X: list of (candidate, features) pair
        """

        self.logger.info("Loading default parameters for Sparse Logistic Regression")
        config = get_config()["learning"]["SparseLogisticRegression"]

        for key in config.keys():
            if key not in self.settings:
                self.settings[key] = config[key]

        # Add one feature for padding vector (all 0s)
        self.settings["input_dim"] = X[1].shape[1] + 1

    def _build_model(self):
        """
        Build the model.
        """

        if "input_dim" not in self.settings:
            raise ValueError("Model parameter input_dim cannot be None.")

        self.sparse_linear = SparseLinear(
            self.settings["input_dim"], self.cardinality, self.settings["bias"]
        )

    def _calc_logits(self, X):
        """
        Calculate the logits.

        :param X: The input data of the model contains features and feature weights.
        :type X: For features: torch.Tensor of shape (batch_size, num_classes) and
            for feature weights: torch.Tensor of shape (batch_size, num_classes)
        :return: The output logits of model.
        :rtype: torch.Tensor of shape (batch_size, num_classes)
        """

        return self.forward(X)
