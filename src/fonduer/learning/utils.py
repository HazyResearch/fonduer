import inspect
import logging
import os
from itertools import product
from multiprocessing import JoinableQueue, Process
from queue import Empty

import numpy as np
import torch
import torch.nn.functional as F
from pandas import DataFrame

from fonduer.learning.models.marginal import Marginal

logger = logging.getLogger(__name__)

# ###########################################################
# # General Learning Utilities
# ###########################################################


def save_marginals(session, X, marginals, training=True):
    """Save marginal probabilities for a set of Candidates to db.

    :param X: A list of arbitrary objects with candidate ids accessible via a
        .id attrib
    :param marginals: A dense M x K matrix of marginal probabilities, where
        K is the cardinality of the candidates, OR a M-dim list/array if K=2.
    :param training: If True, these are training marginals / labels; else they
        are saved as end model predictions.

    Note: The marginals for k=0 are not stored, only for k = 1,...,K
    """
    logger = logging.getLogger(__name__)
    # Make sure that we are working with a numpy array
    try:
        shape = marginals.shape
    except Exception as e:
        marginals = np.array(marginals)
        shape = marginals.shape

    # Handle binary input as M x 1-dim array; assume elements represent
    # poksitive (k=1) class values
    if len(shape) == 1:
        marginals = np.vstack([1 - marginals, marginals]).T

    # Only add values for classes k=1,...,K
    marginal_tuples = []
    for i in range(shape[0]):
        for k in range(1, shape[1] if len(shape) > 1 else 2):
            if marginals[i, k] > 0:
                marginal_tuples.append((i, k, marginals[i, k]))

    # NOTE: This will delete all existing marginals of type `training`
    session.query(Marginal).filter(Marginal.training == training).delete(
        synchronize_session="fetch"
    )

    # Prepare bulk INSERT query
    q = Marginal.__table__.insert()

    # Prepare values
    insert_vals = []
    for i, k, p in marginal_tuples:
        cid = X[i].id
        insert_vals.append(
            {
                "candidate_id": cid,
                "training": training,
                "value": k,
                # We cast p in case its a numpy type, which psycopg2 does not handle
                "probability": float(p),
            }
        )

    # Execute update
    session.execute(q, insert_vals)
    session.commit()
    logger.info("Saved {%d} marginals".format(len(marginals)))


def reshape_marginals(marginals):
    """Returns correctly shaped marginals as np array"""
    # Make sure training marginals are a numpy array first
    try:
        shape = marginals.shape
    except Exception as e:
        marginals = np.array(marginals)
        shape = marginals.shape

    # Set cardinality + marginals in proper format for binary v. categorical
    if len(shape) != 1:
        # If k = 2, make sure is M-dim array
        if shape[1] == 2:
            marginals = marginals[:, 1].reshape(-1)
    return marginals


class LabelBalancer(object):
    def __init__(self, y):
        """Utility class to rebalance training labels.

        :Example:
            To get the indices of a training set with labels y and
            around 90 percent negative examples::

                LabelBalancer(y).get_train_idxs(rebalance=0.1)
        """
        self.y = np.ravel(y)

    def _get_pos(self, split):
        return np.where(self.y > (split + 1e-6))[0]

    def _get_neg(self, split):
        return np.where(self.y < (split - 1e-6))[0]

    def _try_frac(self, m, n, pn):
        # Return (a, b) s.t. a <= m, b <= n
        # and b / a is as close to pn as possible
        r = int(round(float(pn * m) / (1.0 - pn)))
        s = int(round(float((1.0 - pn) * n) / pn))
        return (m, r) if r <= n else ((s, n) if s <= m else (m, n))

    def _get_counts(self, nneg, npos, frac_pos):
        if frac_pos > 0.5:
            return self._try_frac(nneg, npos, frac_pos)
        else:
            return self._try_frac(npos, nneg, 1.0 - frac_pos)[::-1]

    def get_train_idxs(self, rebalance=False, split=0.5, rand_state=None):
        """Get training indices based on y.

        :param rebalance: bool or fraction of positive examples desired If
            True, default fraction is 0.5. If False no balancing.
        :param split: Split point for positive and negative classes.
        """
        rs = np.random if rand_state is None else rand_state
        pos, neg = self._get_pos(split), self._get_neg(split)
        if rebalance:
            if len(pos) == 0:
                raise ValueError("No positive labels.")
            if len(neg) == 0:
                raise ValueError("No negative labels.")
            p = 0.5 if rebalance else rebalance
            n_neg, n_pos = self._get_counts(len(neg), len(pos), p)
            pos = rs.choice(pos, size=n_pos, replace=False)
            neg = rs.choice(neg, size=n_neg, replace=False)
        idxs = np.concatenate([pos, neg])
        rs.shuffle(idxs)
        return idxs


# ##########################################################
# # Advanced Scoring Classes
# ##########################################################


class Scorer(object):
    """Abstract type for scorers"""

    def __init__(self, test_candidates, test_labels, gold_candidate_set=None):
        """
        :param test_candidates: A *list of Candidates* corresponding to
            test_labels
        :param test_labels: A *csrLabelMatrix* of ground truth labels for the
            test candidates
        :param gold_candidate_set: (optional) A *CandidateSet* containing the
            full set of gold labeled candidates
        """
        self.test_candidates = test_candidates
        self.test_labels = test_labels
        self.gold_candidate_set = gold_candidate_set

    def _get_cardinality(self, marginals):
        """Get the cardinality based on the marginals returned by the model."""
        if len(marginals.shape) == 1 or marginals.shape[1] < 3:
            cardinality = 2
        else:
            cardinality = marginals.shape[1]
        return cardinality

    def score(self, test_marginals, **kwargs):
        cardinality = self._get_cardinality(test_marginals)
        if cardinality == 2:
            return self._score_binary(test_marginals, **kwargs)
        else:
            return self._score_categorical(test_marginals, **kwargs)

    def _score_binary(
        self,
        test_marginals,
        train_marginals=None,
        b=0.5,
        set_unlabeled_as_neg=True,
        display=True,
    ):
        raise NotImplementedError()

    def _score_categorical(self, test_marginals, train_marginals=None, display=True):
        raise NotImplementedError()

    def summary_score(self, test_marginals, **kwargs):
        """Return the F1 score (for binary) or accuracy (for categorical)."""
        raise NotImplementedError()


class MentionScorer(Scorer):
    """Scorer for mention level assessment"""

    def _score_binary(
        self,
        test_marginals,
        train_marginals=None,
        b=0.5,
        set_unlabeled_as_neg=True,
        set_at_thresh_as_neg=True,
        display=True,
        **kwargs
    ):
        """
        Return scoring metric for the provided marginals, as well as candidates
        in error buckets.

        :param test_marginals: array of marginals for test candidates
        :param train_marginals (optional): array of marginals for training
            candidates
        :param b: threshold for labeling
        :param set_unlabeled_as_neg: set test labels at the decision threshold
            of b as negative labels
        :param set_at_b_as_neg: set marginals at the decision threshold exactly
            as negative predictions
        :param display: show calibration plots?
        """
        test_label_array = []
        tp = set()
        fp = set()
        tn = set()
        fn = set()

        for i, candidate in enumerate(self.test_candidates):
            # Handle either a LabelMatrix or else assume test_labels array is in
            # correct order i.e. same order as test_candidates
            try:
                test_label_index = self.test_labels.get_row_index(candidate)
                test_label = self.test_labels[test_label_index, 0]
            except AttributeError:
                test_label = self.test_labels[i]

            # Set unlabeled examples to -1 by default
            if test_label == 0 and set_unlabeled_as_neg:
                test_label = -1

            # Bucket the candidates for error analysis
            test_label_array.append(test_label)
            if test_label != 0:
                if test_marginals[i] > b:
                    if test_label == 1:
                        tp.add(candidate)
                    else:
                        fp.add(candidate)
                elif test_marginals[i] < b or set_at_thresh_as_neg:
                    if test_label == -1:
                        tn.add(candidate)
                    else:
                        fn.add(candidate)
        if display:

            # Calculate scores unadjusted for TPs not in our candidate set
            print_scores(
                len(tp), len(fp), len(tn), len(fn), title="Scores (Un-adjusted)"
            )

            # If gold candidate set is provided calculate recall-adjusted scores
            if self.gold_candidate_set is not None:
                gold_fn = [
                    c for c in self.gold_candidate_set if c not in self.test_candidates
                ]
                logger.info("\n")
                print_scores(
                    len(tp),
                    len(fp),
                    len(tn),
                    len(fn) + len(gold_fn),
                    title="Corpus Recall-adjusted Scores",
                )

            # If training and test marginals provided print calibration plots
            if train_marginals is not None and test_marginals is not None:
                raise NotImplementedError("Invalid code here.")
        return tp, fp, tn, fn

    def _score_categorical(
        self, test_marginals, train_marginals=None, display=True, **kwargs
    ):
        """
        Return scoring metric for the provided marginals, as well as candidates
        in error buckets.

        :param test_marginals: array of marginals for test candidates
        :param train_marginals (optional): array of marginals for training
            candidates
        :param display: show calibration plots?
        """
        test_label_array = []
        correct = set()
        incorrect = set()

        # Get predictions
        test_pred = test_marginals.argmax(axis=1) + 1

        # Bucket the candidates for error analysis
        for i, candidate in enumerate(self.test_candidates):
            # Handle either a LabelMatrix or else assume test_labels array is in
            # correct order i.e. same order as test_candidates
            try:
                test_label_index = self.test_labels.get_row_index(candidate)
                test_label = self.test_labels[test_label_index, 0]
            except AttributeError:
                test_label = self.test_labels[i]
            test_label_array.append(test_label)
            if test_label != 0:
                if test_pred[i] == test_label:
                    correct.add(candidate)
                else:
                    incorrect.add(candidate)
        if display:
            nc, ni = len(correct), len(incorrect)
            logger.info("Accuracy: {}".format(nc / float(nc + ni)))

            # If gold candidate set is provided calculate recall-adjusted scores
            if self.gold_candidate_set is not None:
                gold_missed = [
                    c for c in self.gold_candidate_set if c not in self.test_candidates
                ]
                logger.info(
                    "Coverage: {}".format((nc + ni) / (nc + ni + len(gold_missed)))
                )
        return correct, incorrect

    def summary_score(self, test_marginals, **kwargs):
        """
        Return the F1 score (for binary) or accuracy (for categorical).
        Also return the label as second argument.
        """
        error_sets = self.score(test_marginals, display=False, **kwargs)
        if len(error_sets) == 4:
            _, _, f1 = binary_scores_from_counts(*map(len, error_sets))
            return f1, "F1 Score"
        else:
            nc, ninc = map(len, error_sets)
            return nc / float(nc + ninc), "Accuracy"


def binary_scores_from_counts(ntp, nfp, ntn, nfn):
    """Precision, recall, and F1 scores from counts of TP, FP, TN, FN.

    Example usage::

        p, r, f1 = binary_scores_from_counts(*map(len, error_sets))
    """
    prec = ntp / float(ntp + nfp) if ntp + nfp > 0 else 0.0
    rec = ntp / float(ntp + nfn) if ntp + nfn > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1


def print_scores(ntp, nfp, ntn, nfn, title="Scores"):
    prec, rec, f1 = binary_scores_from_counts(ntp, nfp, ntn, nfn)
    pos_acc = ntp / float(ntp + nfn) if ntp + nfn > 0 else 0.0
    neg_acc = ntn / float(ntn + nfp) if ntn + nfp > 0 else 0.0
    logger.info("========================================")
    logger.info(title)
    logger.info("========================================")
    logger.info("Pos. class accuracy: {:.3}".format(pos_acc))
    logger.info("Neg. class accuracy: {:.3}".format(neg_acc))
    logger.info("Precision            {:.3}".format(prec))
    logger.info("Recall               {:.3}".format(rec))
    logger.info("F1                   {:.3}".format(f1))
    logger.info("----------------------------------------")
    logger.info("TP: {} | FP: {} | TN: {} | FN: {}".format(ntp, nfp, ntn, nfn))
    logger.info("========================================\n")


# ##########################################################
# # Grid search
# ##########################################################


class GridSearch(object):
    """
    A class for running a hyperparameter grid search.

    :param model_class: The model class being trained
    :param parameter_dict: A dictionary of (hyperparameter name, list of values)
        pairs. Note that the hyperparameter name must correspond to a keyword
        argument in the `model_class.train` method.
    :param X_train: The training datapoints
    :param Y_train: If applicable, the training labels / marginals
    :param model_class_params: Keyword arguments to pass into model_class
        construction. Note that a new model is constructed for each new
        combination of hyperparameters.
    :param model_hyperparams: Hyperparameters for the model- all must be
            keyword arguments to the `model_class.train` method. Any that are
            included in the grid search will be overwritten.
    :param save_dir: Note that checkpoints will be saved in save_dir/grid_search
    """

    def __init__(
        self,
        model_class,
        parameter_dict,
        X_train,
        Y_train=None,
        model_class_params={},
        model_hyperparams={},
        save_dir="checkpoints",
    ):
        self.model_class = model_class
        self.parameter_dict = parameter_dict
        self.param_names = list(parameter_dict)
        self.X_train = X_train
        self.Y_train = Y_train
        self.model_class_params = model_class_params
        self.model_hyperparams = model_hyperparams
        self.save_dir = os.path.join(save_dir, "grid_search")

    def search_space(self):
        return product(*[self.parameter_dict[pn] for pn in self.param_names])

    def fit(
        self,
        X_valid,
        Y_valid,
        b=0.5,
        beta=1,
        set_unlabeled_as_neg=True,
        n_threads=1,
        eval_batch_size=None,
    ):
        """
        Runs grid search, constructing a new instance of model_class for each
        hyperparameter combination, training on (self.X_train, self.Y_train),
        and validating on (X_valid, Y_valid). Selects the best model according
        to F1 score (binary) or accuracy (categorical).

        :param b: Scoring decision threshold (binary)
        :param beta: F_beta score to select model by (binary)
        :param set_unlabeled_as_neg: Set labels = 0 -> -1 (binary)
        :param n_threads: Parallelism to use for the grid search
        :param eval_batch_size: The batch_size for model evaluation
        """
        if n_threads > 1:
            opt_model, run_stats = self._fit_mt(
                X_valid,
                Y_valid,
                b=b,
                beta=beta,
                set_unlabeled_as_neg=set_unlabeled_as_neg,
                n_threads=n_threads,
                eval_batch_size=eval_batch_size,
            )
        else:
            opt_model, run_stats = self._fit_st(
                X_valid,
                Y_valid,
                b=b,
                beta=beta,
                set_unlabeled_as_neg=set_unlabeled_as_neg,
                eval_batch_size=eval_batch_size,
            )
        return opt_model, run_stats

    def _fit_st(
        self,
        X_valid,
        Y_valid,
        b=0.5,
        beta=1,
        set_unlabeled_as_neg=True,
        eval_batch_size=None,
    ):
        """Single-threaded implementation of `GridSearch.fit`."""
        # Iterate over the param values
        run_stats = []
        run_score_opt = -1.0
        for k, param_vals in enumerate(self.search_space()):
            hps = self.model_hyperparams.copy()

            # Initiate the model from scratch each time
            # Some models may have seed set in the init procedure
            model = self.model_class(**self.model_class_params)
            model_name = "{0}_{1}".format(model.name, k)

            # Set the new hyperparam configuration to test
            for pn, pv in zip(self.param_names, param_vals):
                hps[pn] = pv
            logger.info("=" * 60)
            NUMTYPES = float, int, np.float64
            logger.info(
                "[%d] Testing %s"
                % (
                    k + 1,
                    ", ".join(
                        [
                            "%s = %s"
                            % (pn, ("%0.2e" % pv) if isinstance(pv, NUMTYPES) else pv)
                            for pn, pv in zip(self.param_names, param_vals)
                        ]
                    ),
                )
            )
            logger.info("=" * 60)

            # Train the model
            train_args = [self.X_train]
            if self.Y_train is not None:
                train_args.append(self.Y_train)

            # Pass in the dev set to the train method if applicable, for dev set
            # score printing, best-score checkpointing
            # Note: Need to set the save directory since passing in
            # (X_dev, Y_dev) will by default trigger checkpoint saving
            try:
                model.train(
                    *train_args,
                    X_dev=X_valid,
                    Y_dev=Y_valid,
                    save_dir=self.save_dir,
                    **hps
                )
            except Exception as e:
                model.train(*train_args, **hps)

            # Test the model
            run_scores = model.score(
                X_valid,
                Y_valid,
                b=b,
                beta=beta,
                set_unlabeled_as_neg=set_unlabeled_as_neg,
                batch_size=eval_batch_size,
            )
            if model.cardinality > 2:
                run_score, run_score_label = run_scores, "Accuracy"
                run_scores = [run_score]
            else:
                run_score = run_scores[-1]
                run_score_label = "F-{0} Score".format(beta)

            # Add scores to running stats, print, and set as optimal if best
            logger.info("[{0}] {1}: {2}".format(model.name, run_score_label, run_score))
            run_stats.append(list(param_vals) + list(run_scores))
            if run_score > run_score_opt or k == 0:
                model.save(model_name=model_name, save_dir=self.save_dir)
                # Also save a separate file for easier access
                model.save(
                    model_name="{0}_best".format(model.name), save_dir=self.save_dir
                )
                opt_model_name = model_name
                run_score_opt = run_score

        # Set optimal parameter in the learner model
        opt_model = self.model_class(**self.model_class_params)
        opt_model.load(opt_model_name, save_dir=self.save_dir)

        # Return optimal model & DataFrame of scores
        f_score = "F-{0}".format(beta)
        run_score_labels = (
            ["Acc."] if opt_model.cardinality > 2 else ["Prec.", "Rec.", f_score]
        )
        sort_by = "Acc." if opt_model.cardinality > 2 else f_score
        self.results = DataFrame.from_records(
            run_stats, columns=self.param_names + run_score_labels
        ).sort_values(by=sort_by, ascending=False)
        return opt_model, self.results

    def _fit_mt(
        self,
        X_valid,
        Y_valid,
        b=0.5,
        beta=1,
        set_unlabeled_as_neg=True,
        n_threads=2,
        eval_batch_size=None,
    ):
        """Multi-threaded implementation of `GridSearch.fit`."""
        # First do a preprocessing pass over the data to make sure it is all
        # non-lazily loaded
        # TODO: Better way to go about it than this!!
        logger.info("Loading data...")
        model = self.model_class(**self.model_class_params)
        model._preprocess_data(self.X_train)
        model._preprocess_data(X_valid)

        # Create queue of hyperparameters to test
        logger.info("Launching jobs...")
        params_queue = JoinableQueue()
        param_val_sets = []
        for k, param_vals in enumerate(self.search_space()):
            param_val_sets.append(param_vals)
            hps = self.model_hyperparams.copy()
            for pn, pv in zip(self.param_names, param_vals):
                hps[pn] = pv
            params_queue.put((k, hps))

        # Create a queue to store output results
        scores_queue = JoinableQueue()

        # Start UDF Processes
        ps = []
        for i in range(n_threads):
            p = ModelTester(
                self.model_class,
                self.model_class_params,
                params_queue,
                scores_queue,
                self.X_train,
                X_valid,
                Y_valid,
                Y_train=self.Y_train,
                b=b,
                save_dir=self.save_dir,
                set_unlabeled_as_neg=set_unlabeled_as_neg,
                eval_batch_size=eval_batch_size,
            )
            p.start()
            ps.append(p)

        # Collect scores
        run_stats = []
        while any([p.is_alive() for p in ps]):
            while True:
                try:
                    scores = scores_queue.get(True, QUEUE_TIMEOUT)
                    k = scores[0]
                    param_vals = param_val_sets[k]
                    run_stats.append([k] + list(param_vals) + list(scores[1:]))
                    logger.info("Model {0} Done; score: {1}".format(k, scores[-1]))
                    scores_queue.task_done()
                except Empty:
                    break

        # Terminate the processes
        for p in ps:
            p.terminate()

        # Load best model; first element in each row of run_stats is the model
        # index, last one is the score to sort by
        # Note: the models may be returned out of order!
        i_opt = np.argmax([s[-1] for s in run_stats])
        k_opt = run_stats[i_opt][0]
        model = self.model_class(**self.model_class_params)
        model.load("{0}_{1}".format(model.name, k_opt), save_dir=self.save_dir)

        # Also save the best model as separate file
        model.save(model_name="{0}_best".format(model.name), save_dir=self.save_dir)

        # Return model and DataFrame of scores
        # Test for categorical vs. binary in hack-ey way for now...
        f_score = "F-{0}".format(beta)
        categorical = len(scores) == 2
        labels = ["Acc."] if categorical else ["Prec.", "Rec.", f_score]
        sort_by = "Acc." if categorical else f_score
        self.results = DataFrame.from_records(
            run_stats, columns=["Model"] + self.param_names + labels
        ).sort_values(by=sort_by, ascending=False)
        return model, self.results


QUEUE_TIMEOUT = 3


class ModelTester(Process):
    def __init__(
        self,
        model_class,
        model_class_params,
        params_queue,
        scores_queue,
        X_train,
        X_valid,
        Y_valid,
        Y_train=None,
        b=0.5,
        beta=1,
        set_unlabeled_as_neg=True,
        save_dir="checkpoints",
        eval_batch_size=None,
    ):
        Process.__init__(self)
        self.model_class = model_class
        self.model_class_params = model_class_params
        self.params_queue = params_queue
        self.scores_queue = scores_queue
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        self.scorer_params = {
            "b": b,
            "beta": beta,
            "set_unlabeled_as_neg": set_unlabeled_as_neg,
            "batch_size": eval_batch_size,
        }
        self.save_dir = save_dir

    def run(self):
        while True:
            # Get a new configuration from the queue
            try:
                k, hps = self.params_queue.get(True, QUEUE_TIMEOUT)

                # Initiate the model from scratch each time
                # Some models may have seed set in the init procedure
                model = self.model_class(**self.model_class_params)
                model_name = "{0}_{1}".format(model.name, k)

                # Pass in the dev set to the train method if applicable, for dev
                # set score printing, best-score checkpointing
                if "X_dev" in inspect.getargspec(model.train):
                    hps["X_dev"] = self.X_valid
                    hps["Y_dev"] = self.Y_valid

                # Train model with given hyperparameters
                if self.Y_train is not None:
                    model.train(self.X_train, self.Y_train, **hps)
                else:
                    model.train(self.X_train, **hps)

                # Save the model
                # NOTE: Currently, we have to save every model because we are
                # testing asynchronously. This is obviously memory inefficient,
                # although probably not that much of a problem in practice...
                model.save(model_name=model_name, save_dir=self.save_dir)

                # Test the model
                run_scores = model.score(
                    self.X_valid, self.Y_valid, **self.scorer_params
                )
                run_scores = [run_scores] if model.cardinality > 2 else list(run_scores)

                # Append score to out queue
                self.scores_queue.put([k] + run_scores, True, QUEUE_TIMEOUT)
            except Empty:
                break


class RandomSearch(GridSearch):
    """
    A GridSearch over a random subsample of the hyperparameter search space.

    :param seed: A seed for the GridSearch instance
    """

    def __init__(
        self,
        model_class,
        parameter_dict,
        X_train,
        Y_train=None,
        n=10,
        model_class_params={},
        model_hyperparams={},
        seed=123,
        save_dir="checkpoints",
    ):
        """Search a random sample of size n from a parameter grid"""
        self.rand_state = np.random.RandomState()
        self.rand_state.seed(seed)
        self.n = n
        super(RandomSearch, self).__init__(
            model_class,
            parameter_dict,
            X_train,
            Y_train=Y_train,
            model_class_params=model_class_params,
            model_hyperparams=model_hyperparams,
            save_dir=save_dir,
        )

    def search_space(self):
        return list(
            zip(
                *[
                    self.rand_state.choice(self.parameter_dict[pn], self.n)
                    for pn in self.param_names
                ]
            )
        )


# ##########################################################
# # Loss functions
# ##########################################################


def SoftCrossEntropyLoss(input, target):
    """
    Calculate the CrossEntropyLoss with soft targets

    :param input: prediction logicts
    :param target: target probabilities
    """
    total_loss = torch.tensor(0.0)
    for i in range(input.size(1)):
        cls_idx = torch.full((input.size(0),), i, dtype=torch.long)
        loss = F.cross_entropy(input, cls_idx, reduce=False)
        total_loss += target[:, i].dot(loss)
    return total_loss / input.shape[0]
