import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import issparse

from fonduer.learning.disc_learning import NoiseAwareModel


class LogisticRegression(NoiseAwareModel):
    def forward(self, x):
        """
        Run forward pass.

        :param x: The input (batch) of the model
        """
        return self.linear(x)

    def _check_input(self, X):
        """
        Check input format.

        :param X: The input data of the model
        """
        return isinstance(X, tuple)

    def _preprocess_data(self, X, Y=None, idxs=None, train=False):
        """
        Preprocess the data:
        1. Convert sparse matrix to dense matrix.
        2. Update the order of candidates based on feature index.
        3. Select subset of the input if idxs exists.

        :param X: The input data of the model
        :param X: The labels of input data
        """
        C, F = X
        if issparse(F):
            id2id = dict()
            for i in range(F.shape[0]):
                id2id[F.row_index[i]] = i

            C_ = [None] * len(C)
            for c in C:
                C_[id2id[c.id]] = c

            F = F.todense()

        if idxs is None:
            if Y is not None:
                return [(C_[i], F[i]) for i in range(len(C_))], Y
            else:
                return [(C_[i], F[i]) for i in range(len(C_))]
        if Y is not None:
            return [(C_[i], F[i]) for i in idxs], Y[idxs]
        else:
            return [(C_[i], F[i]) for i in idxs]

    def _update_kwargs(self, X, **model_kwargs):
        """
        Update the model argument.

        :param X: The input data of the model
        :param model_kwargs: The arguments of the model
        """
        model_kwargs["input_dim"] = X[1].shape[1]
        return model_kwargs

    def _build_model(self, model_kwargs):
        """
        Build the model.

        :param model_kwargs: The arguments of the model
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
        :param batch_size: The batch size
        """
        # Generate multi-modal feature input
        F = np.array(list(zip(*X))[1])
        F = torch.Tensor(F).squeeze(1)

        outputs = (
            torch.Tensor([]).cuda()
            if self.model_kwargs["host_device"] in self.gpu
            else torch.Tensor([])
        )

        n = len(F)
        if batch_size is None:
            batch_size = n
        for batch_st in range(0, n, batch_size):
            batch_ed = batch_st + batch_size if batch_st + batch_size <= n else n

            features = (
                F[batch_st:batch_ed].cuda()
                if self.model_kwargs["host_device"] in self.gpu
                else F[batch_st:batch_ed]
            )

            output = self.forward(features)
            if self.cardinality == 2:
                outputs = torch.cat((outputs, output.view(-1)), 0)
            else:
                outputs = torch.cat((outputs, output), 0)

        return outputs
