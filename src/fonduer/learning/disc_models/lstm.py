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
    """
    LSTM model.

    :param name: User-defined name of the model
    :type name: str
    """

    def forward(self, x, f):
        """Forward function.

        :param x: The sequence input (batch) of the model.
        :type x: list of torch.Tensor of shape (sequence_len * batch_size)
        :param f: The feature input of the model.
        :type f: torch.Tensor of shape (batch_size * feature_size)
        :return: The output of LSTM layer.
        :rtype: torch.Tensor of shape (batch_size, num_classes)
        """

        batch_size = len(f)

        outputs = (
            torch.Tensor([]).cuda()
            if self.settings["host_device"] in self._gpu
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
        1. Convert sparse feature matrix to dense matrix for pytorch operation.
        2. Make sentence with mention into sequence data for LSTM.
        3. Select subset of the input if idxs exists.

        :param X: The input data of the model.
        :type X: pair with candidates and corresponding features
        :param Y: The labels of input data (optional).
        :type Y: list of floats if num_classes = 2
            otherwise num_classes-length numpy array
        :param idxs: The selected indexs of input data.
        :type idxs: list or numpy.array
        :param train: An indicator for word dictionary to extend new words.
        :type train: bool
        :return: Preprocessed data.
        :rtype: list of (candidate, features) pairs
        """

        C, F = X

        # Covert sparse feature matrix to dense matrix
        # TODO: the pytorch implementation is taking dense vector as input,
        # should optimize later
        if issparse(F):
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
        for candidate in C:
            cand_idx = []
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

    def _update_settings(self, X):
        """
        Update the model argument.

        :param X: The input data of the model.
        :type X: list of (candidate, features) pairs
        """

        self.logger.info("Load defalut parameters for LSTM")
        config = get_config()["learning"]["LSTM"]

        for key in config.keys():
            if key not in self.settings:
                self.settings[key] = config[key]

        self.settings["relation_arity"] = len(X[0][0])
        self.settings["input_dim"] = X[1].shape[1] + len(X[0][0]) * self.settings[
            "hidden_dim"
        ] * (2 if self.settings["bidirectional"] else 1)

    def _build_model(self):
        """
        Build the model.
        """
        # Set up LSTM modules
        self.lstms = nn.ModuleList(
            [
                RNN(
                    num_classes=0,
                    num_tokens=self.word_dict.s,
                    emb_size=self.settings["emb_dim"],
                    lstm_hidden=self.settings["hidden_dim"],
                    attention=self.settings["attention"],
                    dropout=self.settings["dropout"],
                    bidirectional=self.settings["bidirectional"],
                    use_cuda=self.settings["host_device"] in self._gpu,
                )
            ]
            * self.settings["relation_arity"]
        )

        if "input_dim" not in self.settings:
            raise ValueError("Model parameter input_dim cannot be None.")

        # Set up final linear layer
        self.linear = nn.Linear(
            self.settings["input_dim"], self.cardinality if self.cardinality > 2 else 1
        )

    def _calc_logits(self, X, batch_size=None):
        """
        Calculate the logits.

        :param X: The input data of the model.
        :type X: list of (candidate, features) pairs
        :param batch_size: The batch size.
        :type batch_size: int
        :return: The output logits of model.
        :rtype: torch.Tensor of shape (batch_size, num_classes) if num_classes > 2
            otherwise shape (batch_size, 1)
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
            if self.settings["host_device"] in self._gpu
            else torch.Tensor([])
        )

        n = len(F)
        if batch_size is None:
            batch_size = n
        for batch_st in range(0, n, batch_size):
            batch_ed = batch_st + batch_size if batch_st + batch_size <= n else n

            # TODO: optimize this
            sequences = []
            # For loop each relation arity
            for i in range(len(C[0])):
                sequence = []
                # Generate sequence for the batch
                for j in range(batch_st, batch_ed):
                    sequence.append(C[j][i])
                x, x_mask = pad_batch(sequence, self.settings["max_sentence_length"])
                if self.settings["host_device"] in self._gpu:
                    x = x.cuda()
                    x_mask = x_mask.cuda()
                sequences.append((x, x_mask))

            features = (
                F[batch_st:batch_ed].cuda()
                if self.settings["host_device"] in self._gpu
                else F[batch_st:batch_ed]
            )

            output = self.forward(sequences, features)
            if self.cardinality == 2:
                outputs = torch.cat((outputs, output.view(-1)), 0)
            else:
                outputs = torch.cat((outputs, output), 0)

        return outputs
