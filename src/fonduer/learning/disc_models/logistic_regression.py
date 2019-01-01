import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import issparse

from fonduer.learning.classifier import Classifier
from fonduer.utils.config import get_config


class LogisticRegression(Classifier):
    """
    Logistic Regression model.

    :param name: User-defined name of the model
    :type name: str
    """

    def forward(self, X):
        """Forward function.

        :param X: The input (batch) of the model contains features.
        :type X: torch.Tensor of shape (batch_size, feature_size).
        :return: The output of Logistic Regression layer.
        :rtype: torch.Tensor of shape (batch_size, num_classes)
        """

        return self.linear(X)

    def _check_input(self, X):
        """Check input format.

        :param X: The input data of the model.
        :type X: (candidates, features) pair
        :return: True if valid, otherwise False.
        :rtype: bool
        """

        return isinstance(X, tuple)

    def _preprocess_data(self, X, Y=None, idxs=None, train=False):
        """
        Preprocess the data:
        1. Convert sparse matrix to dense matrix.
        2. Select subset of the input if idxs exists.

        :param X: The input data of the model.
        :type X: pair with candidates and corresponding features
        :param Y: The labels of input data.
        :type Y: list or numpy.array
        :param idxs: The selected indices of input data.
        :type idxs: list or numpy.array
        :param train: Indicator of training set.
        :type train: bool
        :return: Preprocessed data.
        :rtype: list of features
        """

        C, F = X
        if issparse(F):
            F = np.array(F.todense(), dtype=np.float32)

        if Y is not None:
            Y = np.array(Y).astype(np.float32)

        if idxs is None:
            if Y is not None:
                return F, Y
            else:
                return F
        if Y is not None:
            return F[idxs], Y[idxs]
        else:
            return F[idxs]

    def _collate(self, batch):
        """
        Puts each data field into a tensor.

        :param batch: The input data batch.
        :type batch: list of features
        :return: Preprocessed data.
        :rtype: torch.Tensor or pair of torch.Tensor
        """

        if isinstance(batch[0], tuple):
            return [self._cuda(torch.Tensor(samples)) for samples in list(zip(*batch))]
        else:
            return self._cuda(torch.Tensor(batch))

    def _update_settings(self, X):
        """
        Update the model argument.

        :param X: The input data of the model.
        :type X: list of (candidate, features) pair
        """

        self.logger.info("Loading default parameters for Logistic Regression")
        config = get_config()["learning"]["LogisticRegression"]

        for key in config.keys():
            if key not in self.settings:
                self.settings[key] = config[key]

        self.settings["input_dim"] = X[1].shape[1]

    def _build_model(self):
        """
        Build model.
        """

        if "input_dim" not in self.settings:
            raise ValueError("Model parameter input_dim cannot be None.")

        self.linear = nn.Linear(
            self.settings["input_dim"], self.cardinality, self.settings["bias"]
        )

    def _calc_logits(self, X):
        """
        Calculate the logits.

        :param x: The input (batch) of the model.
        :type x: torch.Tensor of shape (batch_size, num_classes)
        :return: The output logits of model.
        :rtype: torch.Tensor of shape (batch_size, num_classes)
        """

        return self.forward(X)
