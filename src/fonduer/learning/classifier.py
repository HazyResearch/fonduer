import numpy as np

from fonduer.learning.utils import save_marginals


class Classifier(object):
    """An abstract class for a probabilistic classifier.

    :param name: Name of the model
    :type name: str
    """

    def __init__(self, name=None):
        self.name = name or self.__class__.__name__
        self.cardinality = 2

    def marginals(self, X, batch_size=None):
        """Calculate the predicted marginal probabilities for the Candidates X.

        :param X: Input data.
        :param batch_size: Batch size.
        :type batch_size: int
        """

        raise NotImplementedError()

    def save_marginals(self, session, X, training=False):
        """Save the predicted marginal probabilities for the Candidates X.

        :param session: The database session to use.
        :param X: Input data.
        :param training: If True, these are training marginals / labels;
            else they are saved as end model predictions.
        :type training: bool
        """

        save_marginals(session, X, self.marginals(X), training=training)

    def predictions(self, X, b=0.5, batch_size=None):
        """Return numpy array of elements in {-1,0,1}
        based on predicted marginal probabilities.

        :param X: Input data.
        :param b: Prediction threshold.
        :type b: float
        :param batch_size: Batch size.
        :type batch_size: int
        """
        if self._check_input(X):
            X = self._preprocess_data(X)

        if self.cardinality > 2:
            return self.marginals(X, batch_size=batch_size).argmax(axis=1) + 1
        else:
            return np.array(
                [
                    1 if p > b else -1 if p < b else 0
                    for p in self.marginals(X, batch_size=batch_size)
                ]
            )

    def score(
        self, X_test, Y_test, b=0.5, set_unlabeled_as_neg=True, beta=1, batch_size=None
    ):
        """
        Returns the summary scores:
            * For binary: precision, recall, F-beta score
            * For categorical: accuracy

        :param X_test: The input test candidates.
        :type X_test: pair with candidates and corresponding features
        :param Y_test: The input test labels.
        :type Y_test: list of labels
        :param b: Decision boundary *for binary setting only*.
        :type b: float
        :param set_unlabeled_as_neg: Whether to map 0 labels -> -1,
            *binary setting.*
        :type set_unlabeled_as_neg: bool
        :param beta: For F-beta score; by default beta = 1 => F-1 score.
        :type beta: int
        :param batch_size: Batch size.
        :type batch_size: int
        """

        if self._check_input(X_test):
            X_test, Y_test = self._preprocess_data(X_test, Y_test)

        predictions = self.predictions(X_test, b=b, batch_size=batch_size)

        # Convert Y_test to dense numpy array
        try:
            Y_test = np.array(Y_test.todense()).reshape(-1)
        except Exception:
            Y_test = np.array(Y_test)

        # Compute accuracy for categorical, or P/R/F1 for binary settings
        if self.cardinality > 2:
            # Compute and return accuracy
            correct = np.where([predictions == Y_test])[0].shape[0]
            return correct / float(Y_test.shape[0])
        else:
            # Either remap or filter out unlabeled (0-valued) test labels
            if set_unlabeled_as_neg:
                Y_test[Y_test == 0] = -1
            else:
                predictions = predictions[Y_test != 0]
                Y_test = Y_test[Y_test != 0]

            # Compute and return precision, recall, and F1 score
            tp = (0.5 * (predictions * Y_test + 1))[predictions == 1].sum()
            pred_pos = predictions[predictions == 1].sum()
            p = tp / float(pred_pos) if pred_pos > 0 else 0.0
            pos = Y_test[Y_test == 1].sum()
            r = tp / float(pos) if pos > 0 else 0.0

            # Compute general F-beta score
            if p + r > 0:
                f_beta = (1 + beta ** 2) * ((p * r) / (((beta ** 2) * p) + r))
            else:
                f_beta = 0.0
            return p, r, f_beta

    def _check_input(self, X):
        """Generic data checking function

        :param X: The input data of the model.
        :type X: pair with candidates and corresponding features
        """
        return True

    def _preprocess_data(self, X, Y=None):
        """Generic preprocessing function

        :param X: The input data of the model.
        :type X: pair with candidates and corresponding features
        :param Y: The labels of input data (optional).
        :type Y: list of labels
        """
        if Y is None:
            return X
        return X, Y

    def save(self):
        """Generic model saving function"""
        raise NotImplementedError()

    def load(self):
        """Generic model loading function"""
        raise NotImplementedError()
