import numpy as np
import torch

from fonduer.learning.disc_learning import NoiseAwareModel
from fonduer.learning.disc_models.layers.sparse_linear import SparseLinear
from fonduer.learning.disc_models.utils import pad_batch


class SparseLogisticRegression(NoiseAwareModel):
    def forward(self, x, w):
        """
        Run forward pass.

        :param x: The input (batch) of the model
        """
        return self.sparse_linear(x, w)

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

        id2id = dict()
        for i in range(F.shape[0]):
            id2id[F.row_index[i]] = i

        C_ = [None] * len(C)
        for c in C:
            C_[id2id[c.id]] = c

        if idxs is None:
            if Y is not None:
                return (
                    [
                        (
                            C_[i],
                            F.indices[F.indptr[i] : F.indptr[i + 1]],
                            F.data[F.indptr[i] : F.indptr[i + 1]],
                        )
                        for i in range(len(C_))
                    ],
                    Y,
                )
            else:
                return [
                    (
                        C_[i],
                        F.indices[F.indptr[i] : F.indptr[i + 1]],
                        F.data[F.indptr[i] : F.indptr[i + 1]],
                    )
                    for i in range(len(C_))
                ]
        if Y is not None:
            return (
                [
                    (
                        C_[i],
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
                    C_[i],
                    F.indices[F.indptr[i] : F.indptr[i + 1]],
                    F.data[F.indptr[i] : F.indptr[i + 1]],
                )
                for i in idxs
            ]

    def _update_kwargs(self, X, **model_kwargs):
        """
        Update the model argument.

        :param X: The input data of the model
        :param model_kwargs: The arguments of the model
        """
        # Add one feature for padding vector (all 0s)
        model_kwargs["input_dim"] = X[1].shape[1] + 1
        return model_kwargs

    def _build_model(self, model_kwargs):
        """
        Build the model.

        :param model_kwargs: The arguments of the model
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
        :param batch_size: The batch size
        """
        # Generate sparse multi-modal feature input
        F = np.array(list(zip(*X))[1]) + 1  # Correct the index since 0 is the padding
        V = np.array(list(zip(*X))[2])

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

            features, _ = pad_batch(F[batch_st:batch_ed], 0)
            values, _ = pad_batch(V[batch_st:batch_ed], 0, type="float")

            if self.model_kwargs["host_device"] in self.gpu:
                features = features.cuda()
                values = values.cuda()

            output = self.forward(features, values)
            if self.cardinality == 2:
                outputs = torch.cat((outputs, output.view(-1)), 0)
            else:
                outputs = torch.cat((outputs, output), 0)

        return outputs
