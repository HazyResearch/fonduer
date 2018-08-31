import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import issparse

from fonduer.learning.disc_learning import NoiseAwareModel
from fonduer.learning.disc_models.layers.rnn import RNN
from fonduer.learning.disc_models.utils import (
    SymbolTable,
    mark_sentence,
    mention_to_tokens,
    pad_batch,
)
from fonduer.utils.config import get_config


class LSTM(NoiseAwareModel):
    def forward(self, x, f):
        """
        Run forward pass.

        :param x: The sequence input (batch) of the model
        :param f: The feature input of the model
        """
        batch_size = len(f)

        outputs = (
            torch.Tensor([]).cuda()
            if self.model_kwargs["host_device"] in self.gpu
            else torch.Tensor([])
        )

        # Calculate textual features from LSTMs
        for i in range(len(x)):
            state_word = self.lstms[0].init_hidden(batch_size)
            output = self.lstms[0].forward(x[i][0], x[i][1], state_word)
            outputs = torch.cat((outputs, output), 1)

        # Concatenate textual features with multi-modal features
        outputs = torch.cat((outputs, f), 1)

        return self.linear(outputs)

    def _check_input(self, X):
        """
        Check input format.

        :param X: The input data of the model
        """
        return isinstance(X, tuple)

    def _preprocess_data(self, X, Y=None, idxs=None, train=False):
        """
        Preprocess the data:
        1. Convert sparse feature matrix to dense matrix for pytorch operation.
        2. Update the order of candidates based on feature index.
        3. Make sentence with mention into sequence data for LSTM.
        4. Select subset of the input if idxs exists.

        :param X: The input data of the model
        :param Y: The labels of input data (optional)
        :param idxs: The selected indexs of input data
        :param train: An indicator for word dictionary to extend new words
        """
        C, F = X

        # Covert sparse feature matrix to dense matrix
        # TODO: the pytorch implementation is taking dense vector as input,
        # should optimize later
        if issparse(F):
            id2id = dict()
            for i in range(F.shape[0]):
                id2id[F.row_index[i]] = i

            C_ = [None] * len(C)
            for c in C:
                C_[id2id[c.id]] = c

            F = F.todense()

        # Create word dictionary for LSTM
        if not hasattr(self, "word_dict"):
            self.word_dict = SymbolTable()
            arity = len(C[0])
            # Add paddings into word dictionary
            for i in range(arity):
                # TODO: optimize this
                list(map(self.word_dict.get, ["~~[[" + str(i), str(i) + "]]~~"]))

        # Make sequence input for LSTM from candidates
        seq_data = []
        for candidate in C_:
            cand_idx = []
            for i in range(len(candidate)):
                # Add mark for each mention in the original sentence
                args = [
                    (
                        candidate[i].span.get_word_start(),
                        candidate[i].span.get_word_end(),
                        i,
                    )
                ]
                s = mark_sentence(mention_to_tokens(candidate[i]), args)
                f = self.word_dict.get if train else self.word_dict.lookup
                cand_idx.append(list(map(f, s)))
            seq_data.append(cand_idx)

        # Generate proprcessed the input
        if idxs is None:
            if Y is not None:
                return [(seq_data[i], F[i]) for i in range(len(seq_data))], Y
            else:
                return [(seq_data[i], F[i]) for i in range(len(seq_data))]
        if Y is not None:
            return [(seq_data[i], F[i]) for i in idxs], Y[idxs]
        else:
            return [(seq_data[i], F[i]) for i in idxs]

    def _update_kwargs(self, X, **model_kwargs):
        """
        Update the model argument.

        :param X: The input data of the model
        :param model_kwargs: The arguments of the model
        """
        self.logger.info("Load defalut parameters for LSTM")
        settings = get_config()["learning"]["LSTM"]

        for key in settings.keys():
            if key not in model_kwargs:
                model_kwargs[key] = settings[key]

        model_kwargs["relation_arity"] = len(X[0][0])
        model_kwargs["input_dim"] = X[1].shape[1] + len(X[0][0]) * model_kwargs[
            "hidden_dim"
        ] * (2 if model_kwargs["bidirectional"] else 1)

        return model_kwargs

    def _build_model(self, model_kwargs):
        """
        Build the model.

        :param model_kwargs: The arguments of the model
        """
        # Set up LSTM modules
        self.lstms = nn.ModuleList(
            [
                RNN(
                    n_classes=0,
                    num_tokens=self.word_dict.s,
                    emb_size=model_kwargs["emb_dim"],
                    lstm_hidden=model_kwargs["hidden_dim"],
                    attention=model_kwargs["attention"],
                    dropout=model_kwargs["dropout"],
                    bidirectional=model_kwargs["bidirectional"],
                    use_cuda=model_kwargs["host_device"] in self.gpu,
                )
            ]
            * model_kwargs["relation_arity"]
        )

        if "input_dim" not in model_kwargs:
            raise ValueError("Kwarg input_dim cannot be None.")

        # Set up final linear layer
        self.linear = nn.Linear(
            model_kwargs["input_dim"], self.cardinality if self.cardinality > 2 else 1
        )

    def _calc_logits(self, X, batch_size=None):
        """
        Calculate the logits.

        :param X: The input data of the model
        :param batch_size: The batch size
        """
        # Generate LSTM input
        C = np.array(list(zip(*X))[0])

        # Check LSTM input dimension size matches the number of lstms in the model
        assert len(C[0]) == len(self.lstms)

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

            # TODO: optimize this
            sequences = []
            for i in range(len(C[0])):
                sequence = []
                for j in range(batch_st, batch_ed):
                    sequence.append(C[j][i])
                x, x_mask = pad_batch(
                    sequence, self.model_kwargs["max_sentence_length"]
                )
                if self.model_kwargs["host_device"] in self.gpu:
                    x = x.cuda()
                    x_mask = x_mask.cuda()
                sequences.append((x, x_mask))

            features = (
                F[batch_st:batch_ed].cuda()
                if self.model_kwargs["host_device"] in self.gpu
                else F[batch_st:batch_ed]
            )

            output = self.forward(sequences, features)
            if self.cardinality == 2:
                outputs = torch.cat((outputs, output.view(-1)), 0)
            else:
                outputs = torch.cat((outputs, output), 0)

        return outputs
