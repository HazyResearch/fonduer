import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import issparse

from fonduer.learning.disc_learning import NoiseAwareModel


class LogisticRegression(NoiseAwareModel):
    def forward(self, x):
        """Forward function.

        :param x: The input (batch) of the model
        :type x: torch.Tensor
        """

        return self.linear(x)

    def _check_input(self, X):
        """Check input format.

        :param X: The input data of the model
        :type X: pair
        """

        return isinstance(X, tuple)

    def _preprocess_data(self, X, Y=None, idxs=None, train=False):
        """
        Preprocess the data:
        1. Convert sparse matrix to dense matrix.
        2. Select subset of the input if idxs exists.

        :param X: The input data of the model
        :type X: pair with candidates and corresponding features
        :param Y: The labels of input data
        :type Y: list
        :param idxs: The selected indices of input data
        :type idxs: list or numpy.array
        :param train: Indicator of training set.
        :type train: bool
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

    def _update_kwargs(self, X, **model_kwargs):
        """
        Update the model argument.

        :param X: The input data of the model
        :type X: list
        :param model_kwargs: The arguments of the model
        :type model_kwargs: dict
        """

        model_kwargs["input_dim"] = X[1].shape[1]
        return model_kwargs

    def _build_model(self, model_kwargs):
        """
        Build the model.

        :param model_kwargs: The arguments of the model
        :type model_kwargs: dict
        """

        if "input_dim" not in model_kwargs:
            raise ValueError("Kwarg input_dim cannot be None.")

        self.linear = nn.Linear(
            model_kwargs["input_dim"], self.cardinality if self.cardinality > 2 else 1
        )

    def _calc_logits(self, X, batch_size=None):
        """
        Calculate the logits.

        :param X: The input data of the model
        :type X: list
        :param batch_size: The batch size
        :type batch_size: int
        """

        # Generate multi-modal feature input
        F = np.array(list(zip(*X))[1])
        F = torch.Tensor(F).squeeze(1)

        outputs = (
            torch.Tensor([]).cuda()
            if self.model_kwargs["host_device"] in self._gpu
            else torch.Tensor([])
        )

        n = len(F)
        if batch_size is None:
            batch_size = n
        for batch_st in range(0, n, batch_size):
            batch_ed = batch_st + batch_size if batch_st + batch_size <= n else n

            features = (
                F[batch_st:batch_ed].cuda()
                if self.model_kwargs["host_device"] in self._gpu
                else F[batch_st:batch_ed]
            )

            output = self.forward(features)
            if self.cardinality == 2:
                outputs = torch.cat((outputs, output.view(-1)), 0)
            else:
                outputs = torch.cat((outputs, output), 0)

        return outputs
