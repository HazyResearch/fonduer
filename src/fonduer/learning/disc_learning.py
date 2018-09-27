import logging
import os
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from fonduer.learning.classifier import Classifier
from fonduer.learning.utils import (
    LabelBalancer,
    SoftCrossEntropyLoss,
    reshape_marginals,
)


class NoiseAwareModel(Classifier, nn.Module):
    """
    Generic NoiseAwareModel class.

    :param name: Name of the model
    :type name: str
    """

    _gpu = ["gpu", "GPU"]

    def __init__(self, name=None):
        self.logger = logging.getLogger(__name__)
        self.settings = None
        Classifier.__init__(self, name)
        nn.Module.__init__(self)

    def _set_random_seed(self, seed):
        # Set random seed for all numpy operations
        self.rand_state = np.random.RandomState(seed=seed)
        np.random.seed(seed=seed)

        # Set random seed for PyTorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _build_model(self):
        raise NotImplementedError()

    def _update_settings(self, X):
        pass

    def _preprocess_data(self, X, Y, idxs=None, train=False):
        return X, Y

    def _setup_model_loss(self, lr):
        """
        Setup loss and optimizer for PyTorch model.
        """
        # Setup loss
        if not hasattr(self, "loss"):
            if self.cardinality > 2:
                self.loss = SoftCrossEntropyLoss
            else:
                self.loss = nn.BCEWithLogitsLoss()

        # Setup optimizer
        if not hasattr(self, "optimizer"):
            self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def _check_input(self, X):
        """Checks correctness of input; optional to implement."""
        pass

    def _calc_logits(self, X, batch_size):
        """Calculate the logits of input."""
        raise NotImplementedError()

    def train(
        self,
        X_train,
        Y_train,
        n_epochs=25,
        lr=0.01,
        batch_size=256,
        rebalance=False,
        X_dev=None,
        Y_dev=None,
        print_freq=5,
        dev_ckpt=True,
        dev_ckpt_delay=0.75,
        save_dir="checkpoints",
        seed=1234,
        host_device="CPU",
    ):
        """
        Generic training procedure for PyTorch model

        :param X_train: The training data which is a (list of Candidate objects,
            a sparse matrix of corresponding features) pair.
        :type X_train: pair
        :param Y_train: Array of marginal probabilities for each Candidate.
        :type Y_train: list or numpy.array
        :param n_epochs: Number of training epochs.
        :type n_epochs: int
        :param lr: Learning rate.
        :type lr: float
        :param batch_size: Batch size for learning model.
        :type batch_size: int
        :param rebalance: Bool or fraction of positive examples for training.
                    - if True, defaults to standard 0.5 class balance
                    - if False, no class balancing
        :type rebalance: bool
        :param X_dev: Candidates for evaluation, same format as X_train.
        :param Y_dev: Labels for evaluation, same format as Y_train.
        :param print_freq: number of epochs at which to print status, and if present,
            evaluate the dev set (X_dev, Y_dev).
        :type print_freq: int
        :param dev_ckpt: If True, save a checkpoint whenever highest score
            on (X_dev, Y_dev) reached. Note: currently only evaluates at
            every @print_freq epochs.
        :param dev_ckpt_delay: Start dev checkpointing after this portion
            of n_epochs.
        :type dev_ckpt_delay: float
        :param save_dir: Save dir path for checkpointing.
        :type save_dir: str
        :param seed: Random seed
        :type seed: int
        :param host_device: Host device
        :type host_device: str
        """

        # Set model parameters
        self.settings = {
            "n_epochs": n_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "seed": 1234,
            "host_device": host_device,
        }

        # Set random seed
        self._set_random_seed(self.settings["seed"])

        self._check_input(X_train)
        verbose = print_freq > 0

        # Update cardinality of the model with training marginals
        self.cardinality = Y_train.shape[1] if len(Y_train.shape) > 1 else 2

        # Make sure marginals are in correct default format
        Y_train = reshape_marginals(Y_train)
        # Make sure marginals are in [0,1] (v.s e.g. [-1, 1])
        if self.cardinality > 2 and not np.all(Y_train.sum(axis=1) - 1 < 1e-10):
            raise ValueError("Y_train must be row-stochastic (rows sum to 1).")
        if not np.all(Y_train >= 0):
            raise ValueError("Y_train must have values in [0,1].")

        # Remove unlabeled examples and optionally rebalance training set
        # Note: rebalancing only for binary setting currently
        if self.cardinality == 2:
            # This removes unlabeled examples and optionally rebalances
            train_idxs = LabelBalancer(Y_train).get_train_idxs(
                rebalance, rand_state=self.rand_state
            )
        else:
            # In categorical setting, just remove unlabeled
            diffs = Y_train.max(axis=1) - Y_train.min(axis=1)
            train_idxs = np.where(diffs > 1e-6)[0]

        self._update_settings(X_train)

        _X_train, _Y_train = self._preprocess_data(
            X_train, Y_train, idxs=train_idxs, train=True
        )
        if X_dev is not None:
            _X_dev, _Y_dev = self._preprocess_data(X_dev, Y_dev)

        if self.settings["host_device"] in self._gpu:
            if not torch.cuda.is_available():
                self.settings["host_device"] = "CPU"
                self.logger.info("GPU is not available, switching to CPU...")
            else:
                self.logger.info("Using GPU...")

        self.logger.info("Settings: {}".format(self.settings))

        # Build network
        self._build_model()
        self._setup_model_loss(self.settings["lr"])

        # Set up GPU if necessary
        if self.settings["host_device"] in self._gpu:
            nn.Module.cuda(self)

        # Run mini-batch SGD
        n = len(_X_train)
        if self.settings["batch_size"] > n:
            self.logger.info("Switching batch size to {} for training.".format(n))
        batch_size = min(self.settings["batch_size"], n)

        if verbose:
            st = time()
            self.logger.info("[{0}] Training model".format(self.name))
            self.logger.info(
                "[{0}] n_train={1}  #epochs={2}  batch size={3}".format(
                    self.name, n, self.settings["n_epochs"], batch_size
                )
            )

        dev_score_opt = 0.0
        dev_score_epo = -1
        for epoch in range(self.settings["n_epochs"]):
            iteration_losses = []

            # Shuffle the training data
            idxs = self.rand_state.permutation(n)

            nn.Module.train(self, True)
            for batch_st in range(0, n, batch_size):
                # zero gradients for each batch
                self.optimizer.zero_grad()

                batch_ed = batch_st + batch_size if batch_st + batch_size <= n else n

                output = self._calc_logits(
                    [_X_train[idx] for idx in idxs[batch_st:batch_ed]], batch_size
                )

                # Calculate loss for current batch
                y = torch.Tensor(_Y_train[idxs[batch_st:batch_ed]])
                if self.settings["host_device"] in self._gpu:
                    y = y.cuda()
                loss = self.loss(output, y)

                # Compute gradient
                loss.backward()

                # Update the parameters
                self.optimizer.step()

                if self.settings["host_device"] in self._gpu:
                    iteration_losses.append(loss.cpu())
                else:
                    iteration_losses.append(loss)

            # Print training stats and optionally checkpoint model
            if verbose and (
                (
                    (epoch + 1) % print_freq == 0
                    or epoch in [0, (self.settings["n_epochs"] - 1)]
                )
            ):
                msg = "[{0}] Epoch {1} ({2:.2f}s)\tAverage loss={3:.6f}".format(
                    self.name,
                    epoch + 1,
                    time() - st,
                    torch.stack(iteration_losses).mean(),
                )
                if X_dev is not None:
                    scores = self.score(
                        _X_dev, Y_dev, batch_size=self.settings["batch_size"]
                    )
                    score = scores if self.cardinality > 2 else scores[-1]
                    score_label = "Acc." if self.cardinality > 2 else "F1"
                    msg += "\tDev {0}={1:.2f}".format(score_label, 100.0 * score)
                self.logger.info(msg)

                # If best score on dev set so far and dev checkpointing is
                # active, save checkpoint
                if (
                    X_dev is not None
                    and dev_ckpt
                    and epoch > dev_ckpt_delay * self.settings["n_epochs"]
                    and score > dev_score_opt
                ):
                    dev_score_opt = score
                    dev_score_epo = epoch
                    self.save(save_dir=save_dir, global_step=dev_score_epo)

        # Conclude training
        if verbose:
            self.logger.info(
                "[{0}] Training done ({1:.2f}s)".format(self.name, time() - st)
            )
        # If checkpointing on, load last checkpoint (i.e. best on dev set)
        if dev_ckpt and X_dev is not None and verbose and dev_score_opt > 0:
            self.logger.info("Loading best checkpoint")
            self.load(save_dir=save_dir, global_step=dev_score_epo)

    def marginals(self, X, batch_size=None):
        """
        Compute the marginals for the given candidates X.
        Note: split into batches to avoid OOM errors.

        :param X: The input data which is a (list of Candidate objects, a sparse
            matrix of corresponding features) pair or a list of
            (Candidate, features) pairs.
        :type X: pair or list
        :param batch_size: Batch size.
        :type batch_size: int
        """
        nn.Module.train(self, False)

        if self._check_input(X):
            X = self._preprocess_data(X)

        marginal = self._calc_logits(X, batch_size)

        if self.settings["host_device"] in self._gpu:
            marginal = marginal.cpu()

        if self.cardinality == 2:
            return torch.sigmoid(marginal).detach().numpy()
        else:
            return torch.softmax(marginal).detach().numpy()

    def save(
        self, model_name=None, save_dir="checkpoints", verbose=True, global_step=0
    ):
        """Save current model.

        :param model_name: Saved model name.
        :type model_name: str
        :param save_dir: Saved model directory.
        :type save_dir: str
        :param verbose: Print log or not
        :type verbose: bool
        :param global_step: learned epoch of saved model
        :type global_step: int
        """

        model_name = model_name or self.name

        # Check existence of model saving directory and create if does not exist.
        model_dir = os.path.join(save_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        params = {
            "model": self.state_dict(),
            "cardinality": self.cardinality,
            "name": model_name,
            "config": self.settings,
            "epoch": global_step,
        }

        model_file = "{0}.mdl.ckpt.{1}".format(model_name, global_step)

        try:
            torch.save(params, "{0}/{1}".format(model_dir, model_file))
        except BaseException:
            self.logger.warning("Saving failed... continuing anyway.")

        if verbose:
            self.logger.info(
                "[{0}] Model saved as {1} in {2}".format(
                    model_name, model_file, model_dir
                )
            )

    def load(
        self, model_name=None, save_dir="checkpoints", verbose=True, global_step=0
    ):
        """Load model from file and rebuild the model.

        :param model_name: Saved model name.
        :type model_name: str
        :param save_dir: Saved model directory.
        :type save_dir: str
        :param verbose: Print log or not
        :type verbose: bool
        :param global_step: learned epoch of saved model
        :type global_step: int
        """

        model_name = model_name or self.name

        model_dir = os.path.join(save_dir, model_name)
        if not os.path.exists(model_dir):
            self.logger.error("Loading failed... Directory does not exist.")

        model_file = "{0}.mdl.ckpt.{1}".format(model_name, global_step)

        try:
            checkpoint = torch.load("{0}/{1}".format(model_dir, model_file))
        except BaseException:
            self.logger.error(
                "Loading failed... Cannot load model from {0} in {1}".format(
                    model_name, model_dir
                )
            )

        self.load_state_dict(checkpoint["model"])
        self.settings = checkpoint["config"]
        self.cardinality = checkpoint["cardinality"]
        self.name = checkpoint["name"]

        if verbose:
            self.logger.info(
                "[{0}] Model loaded as {1} in {2}".format(
                    model_name, model_file, model_dir
                )
            )
