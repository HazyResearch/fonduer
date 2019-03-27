import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import issparse

from fonduer.learning.classifier import Classifier
from fonduer.learning.disc_models.modules.rnn import RNN
from fonduer.learning.disc_models.utils import (
    SymbolTable,
    mark_sentence,
    mention_to_tokens,
    pad_batch,
)
from fonduer.utils.config import get_config


class LSTM(Classifier):
    """
    LSTM model.

    :param name: User-defined name of the model
    :type name: str
    """

    def forward(self, X):
        """Forward function.

        :param X: The input (batch) of the model contains word sequences for lstm
            and features.
        :type X: For word sequences: a list of torch.Tensor pair (word sequence
            and word mask) of shape (batch_size, sequence_length).
            For features: torch.Tensor of shape (batch_size, feature_size).
        :return: The output of LSTM layer.
        :rtype: torch.Tensor of shape (batch_size, num_classes)
        """

        s = X[:-1]
        f = X[-1]

        batch_size = len(f)

        outputs = self._cuda(torch.Tensor([]))

        # Calculate textual features from LSTMs
        for i in range(len(s)):
            state_word = self.lstms[0].init_hidden(batch_size)
            output = self.lstms[0].forward(s[i][0], s[i][1], state_word)
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
        :type Y: list or numpy.array
        :param idxs: The selected indexs of input data.
        :type idxs: list or numpy.array
        :param train: An indicator for word dictionary to extend new words.
        :type train: bool
        :return: Preprocessed data.
        :rtype: list of (candidate, features) pairs
        """

        C, F = X

        # Covert sparse feature matrix to dense matrix
        if issparse(F):
            F = np.array(F.todense(), dtype=np.float32)

        if Y is not None:
            Y = np.array(Y).astype(np.float32)

        # Create word dictionary for LSTM
        if not hasattr(self, "word_dict"):
            self.word_dict = SymbolTable()
            arity = len(C[0])
            # Add paddings into word dictionary
            for i in range(arity):
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
                return [[seq_data[i], F[i]] for i in range(len(seq_data))], Y
            else:
                return [[seq_data[i], F[i]] for i in range(len(seq_data))]
        if Y is not None:
            return [[seq_data[i], F[i]] for i in idxs], Y[idxs]
        else:
            return [[seq_data[i], F[i]] for i in idxs]

    def _collate(self, batch):
        """
        Puts each data field into a tensor.

        :param batch: The input data batch.
        :type batch: list of (candidate, features) pairs
        :return: Preprocessed data.
        :rtype: list of torch.Tensor with torch.Tensor (Optional)
        """

        Y_batch = None
        if isinstance(batch[0], tuple):
            batch, Y_batch = list(zip(*batch))
            Y_batch = self._cuda(torch.Tensor(Y_batch))

        batch, f_batch = list(zip(*batch))

        X_batch = []

        for samples in list(zip(*batch)):
            x, x_mask = pad_batch(samples, max_len=self.settings["max_sentence_length"])
            X_batch.append((self._cuda(x), self._cuda(x_mask)))
        X_batch.append(self._cuda(torch.Tensor(f_batch)))

        if Y_batch is not None:
            return X_batch, Y_batch
        else:
            return X_batch

    def _update_settings(self, X):
        """
        Update the model argument.

        :param X: The input data of the model.
        :type X: list of (candidate, features) pairs
        """

        self.logger.info("Loading default parameters for LSTM")
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
        self.linear = nn.Linear(self.settings["input_dim"], self.cardinality)

    def _calc_logits(self, X):
        """
        Calculate the logits.

        :param X: The input (batch) of the model contains word sequences for lstm
            and features.
        :type X: For word sequences: a list of torch.Tensor pair (word sequence
            and word mask) of shape (batch_size, sequence_length).
            For features: torch.Tensor of shape (batch_size, feature_size).
        :return: The output logits of model.
        :rtype: torch.Tensor of shape (batch_size, num_classes)
        """

        return self.forward(X)
