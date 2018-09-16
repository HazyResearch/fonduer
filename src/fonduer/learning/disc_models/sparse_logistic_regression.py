import numpy as np
import torch

from fonduer.learning.disc_learning import NoiseAwareModel
from fonduer.learning.disc_models.layers.sparse_linear import SparseLinear
from fonduer.learning.disc_models.utils import pad_batch


class SparseLogisticRegression(NoiseAwareModel):
    """
    Sparse Logistic Regression model.

    :param seed: Random seed of model which is passed into both numpy and PyTorch.
    :type seed: int
    :param cardinality: Cardinality of class
    :type cardinality: int
    :param name: User-defined name of the model
    :type name: str
    """

    def forward(self, x, w):
        """
        Run forward pass.

        :param x: The input (batch) of the model
        :type x: torch.Tensor
        :return: The output of sparse Logistic Regression layer
        :rtype: torch.Tensor
        """
        return self.sparse_linear(x, w)

    def _check_input(self, X):
        """
        Check input format.

        :param X: The input data of the model
        :type X: pair
        :return: True if valid, otherwise False
        :rtype: bool
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
        :return: Preprocessed data.
        :rtype: list of (candidate, fetures) pair
        """

        C, F = X

        if idxs is None:
            if Y is not None:
                return (
                    [
                        (
                            C[i],
                            F.indices[F.indptr[i] : F.indptr[i + 1]],
                            F.data[F.indptr[i] : F.indptr[i + 1]],
                        )
                        for i in range(len(C))
                    ],
                    Y,
                )
            else:
                return [
                    (
                        C[i],
                        F.indices[F.indptr[i] : F.indptr[i + 1]],
                        F.data[F.indptr[i] : F.indptr[i + 1]],
                    )
                    for i in range(len(C))
                ]
        if Y is not None:
            return (
                [
                    (
                        C[i],
                        F.indices[F.indptr[i] : F.indptr[i + 1]],
                        F.data[F.indptr[i] : F.indptr[i + 1]],
                    )
                    for i in idxs
                ],
                Y[idxs],
            )
        else:
            return [
                (
                    C[i],
                    F.indices[F.indptr[i] : F.indptr[i + 1]],
                    F.data[F.indptr[i] : F.indptr[i + 1]],
                )
                for i in idxs
            ]

    def _update_kwargs(self, X, **model_kwargs):
        """
        Update the model argument.

        :param X: The input data of the model
        :type X: list
        :param model_kwargs: The arguments of the model
        :type model_kwargs: dict
        :return: Updated model arguments
        :rtype: dict
        """

        # Add one feature for padding vector (all 0s)
        model_kwargs["input_dim"] = X[1].shape[1] + 1
        return model_kwargs

    def _build_model(self, model_kwargs):
        """
        Build the model.

        :param model_kwargs: The arguments of the model
        :type model_kwargs: dict
        """

        if "input_dim" not in model_kwargs:
            raise ValueError("Kwarg input_dim cannot be None.")

        cardinality = self.cardinality if self.cardinality > 2 else 1
        bias = False if "bias" not in model_kwargs else model_kwargs["bias"]

        self.sparse_linear = SparseLinear(model_kwargs["input_dim"], cardinality, bias)

    def _calc_logits(self, X, batch_size=None):
        """
        Calculate the logits.

        :param X: The input data of the model
        :type X: list
        :param batch_size: The batch size
        :type batch_size: int
        :return: The output logits of model
        :rtype: torch.Tensor
        """

        # Generate sparse multi-modal feature input
        F = np.array(list(zip(*X))[1]) + 1  # Correct the index since 0 is the padding
        V = np.array(list(zip(*X))[2])

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

            features, _ = pad_batch(F[batch_st:batch_ed], 0)
            values, _ = pad_batch(V[batch_st:batch_ed], 0, type="float")

            if self.model_kwargs["host_device"] in self._gpu:
                features = features.cuda()
                values = values.cuda()

            output = self.forward(features, values)
            if self.cardinality == 2:
                outputs = torch.cat((outputs, output.view(-1)), 0)
            else:
                outputs = torch.cat((outputs, output), 0)

        return outputs
