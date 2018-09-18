import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import issparse

from fonduer.learning.disc_learning import NoiseAwareModel
from fonduer.utils.config import get_config


class LogisticRegression(NoiseAwareModel):
    """
    Logistic Regression model.

    :param name: User-defined name of the model
    :type name: str
    """

    def forward(self, x):
        """Forward function.

        :param x: The input (batch) of the model.
        :type x: torch.Tensor of shape (batch_size, num_classes)
        :return: The output of Logistic Regression layer.
        :rtype: torch.Tensor of shape (batch_size, num_classes)
        """

        return self.linear(x)

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
        :type Y: list of float if num_classes = 2
            otherwise num_classes-length numpy array
        :param idxs: The selected indices of input data.
        :type idxs: list or numpy.array
        :param train: Indicator of training set.
        :type train: bool
        :return: Preprocessed data.
        :rtype: list of (candidate, features) pair
        """

        C, F = X
        if issparse(F):
            F = F.todense()

        if idxs is None:
            if Y is not None:
                return [(C[i], F[i]) for i in range(len(C))], Y
            else:
                return [(C[i], F[i]) for i in range(len(C))]
        if Y is not None:
            return [(C[i], F[i]) for i in idxs], Y[idxs]
        else:
            return [(C[i], F[i]) for i in idxs]

    def _update_settings(self, X):
        """
        Update the model argument.

        :param X: The input data of the model.
        :type X: list of (candidate, features) pair
        """

        self.logger.info("Load defalut parameters for Logistic Regression")
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
            self.settings["input_dim"],
            self.cardinality if self.cardinality > 2 else 1,
            self.settings["bias"],
        )

    def _calc_logits(self, X, batch_size=None):
        """
        Calculate the logits.

        :param X: The input data of the model.
        :type X: list of (candidate, fetures) pair
        :param batch_size: The batch size.
        :type batch_size: int
        :return: The output logits of model.
        :rtype: torch.Tensor of shape (batch_size, num_classes) if num_classes > 2
            otherwise shape (batch_size, 1)
        """

        # Generate multi-modal feature input
        F = np.array(list(zip(*X))[1])
        F = torch.Tensor(F).squeeze(1)

        outputs = (
            torch.Tensor([]).cuda()
            if self.settings["host_device"] in self._gpu
            else torch.Tensor([])
        )

        n = len(F)
        if batch_size is None:
            batch_size = n
        for batch_st in range(0, n, batch_size):
            batch_ed = batch_st + batch_size if batch_st + batch_size <= n else n

            features = (
                F[batch_st:batch_ed].cuda()
                if self.settings["host_device"] in self._gpu
                else F[batch_st:batch_ed]
            )

            output = self.forward(features)
            if self.cardinality == 2:
                outputs = torch.cat((outputs, output.view(-1)), 0)
            else:
                outputs = torch.cat((outputs, output), 0)

        return outputs
