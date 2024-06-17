from abc import ABC, abstractmethod

from sklearn.neighbors import NearestNeighbors

import numpy as np

try:
    import tensorflow as tf
    from pu.losses.pu_losses import create_nn_pu_loss
except:
    pass

class PUAlgorithm(ABC):
    '''
    Learn from positive and unlabeled data.

    This is the superclass for all algorithms that learn from positive
    and unlabeled data.

    The structure of these algorithms is mostly based on the survey
    "Learning From Positive and Unlabeled Data: A Survey", by Jessa Bekker and Jesse Davis.

    Parameters
    ----------
    verbose: whether to print or not debug messages
    '''
    def __init__(self, verbose=False):
        self.verbose = verbose

    @abstractmethod
    def fit(self, X, y, X_val, y_val):
        '''
        Fit given data to perform prediction.

        This abstract method is the core of all PU algorithms, and it must be implemented by all
        subclasses of it. It takes positive and unlabeled data, and trains a model on top of it.
        
        Parameters
        ----------
        X: a numpy array with shape (num_examples, example_size), with the training features.

        y: a numpy array with shape (num_examples), with the labels. They can be 'P' (for positive data),
           or 'U' (for unlabeled data).

        X_val: a numpy array with shape (num_validation_examples, example_size), with the validation
                    examples.

        y_val: a numpy array with shape (num_examples), with the validation labels. They can be 'P' (for positive data),
           or 'U' (for unlabeled data).

        Returns
        -------
        self: this same object
        '''
        pass

    @abstractmethod
    def predict(self, X):
        '''
        Get label predictions for a series of examples with the trained model.

        Parameters
        ----------
        X: the examples to get the predictions for
        '''
        pass

    @abstractmethod
    def predict_proba(self, X):
        '''
        Get label probabilites for a series of examples with the trained model.

        Parameters
        ----------
        X: the examples to get the probabilities for
        '''
        pass

    def print_verbose(self, *args, **kwargs):
        '''
        Wrapper for Python print, which prints only if the verbose flag is set

        Parameters
        ----------
        args and kwargs: arguments to Python print
        '''
        if self.verbose:
            print(*args, **kwargs)

class NonNegativePU(PUAlgorithm):
    '''
    Implements differentiable models that incorporate the loss function described in 
    "Positive-Unlabeled Learning with Non-Negative Risk Estimator". These are specified by a base model,
    , to which this loss function is applied. This model requires TensorFlow.

    Parameters
    ----------
    model: base model for classifying features.
    positive_prior: hand-calculated estimate of the positive class prior. It can be 'auto' for
                    changing the estimate on each epoch, by using the positive rate in the 
                    validation set
    loss_fn: underlying loss function for the PU loss function.
    compile_kwargs: keyword arguments for model.compile.
    fit_kwargs: keyword arguments for model.fit
    verbose: whether to print or not debug messages.
    '''
    def __init__(self, model, positive_prior, loss_fn, compile_kwargs, fit_kwargs, verbose=False):
        super().__init__(verbose)

        self.positive_prior = positive_prior

        self.base_loss = loss_fn
        self.loss_fn = create_nn_pu_loss(0.5 if positive_prior == 'auto' else positive_prior, loss_fn)
        self.model = model

        self.model.compile(loss=self.loss_fn, **compile_kwargs)
        self.compile_kwargs = compile_kwargs
        self.fit_kwargs = fit_kwargs
        self.epochs = fit_kwargs['epochs']

    def fit(self, X, y, X_val=None, y_val=None):
        '''
        Fits the non-negative risk estimator model to the given data. Can accept optional validation
        data for training the underlying neural network model.

        Parameters
        ----------
        X: numpy array with shape (num_examples, example_size), with the training features.
        y: numpy array with shape (num_examples), with the labels. They can be 'P' (for positive data),
           or 'U' (for unlabeled data).
        X_val: validation features (optional in this algorithm)
        y_val: validation labels (optional in this algorithm)
        '''
        val_data = None if X_val is None or y_val is None else (X_val, y_val)
        if self.positive_prior != 'auto':
            self.model.fit(X, y, validation_data = val_data, **self.fit_kwargs)

        else:
            # Change positive prior estimate if configured to do so
            self.fit_kwargs.pop('epochs', None)
            for i in range(self.epochs):
                self.model.fit(X, y, validation_data = val_data, **self.fit_kwargs, epochs=1)
                preds = self.predict(X_val)
                new_prior = np.sum(preds) / preds.shape[0]
                self.loss_fn = create_nn_pu_loss(new_prior, self.base_loss)
                self.model.compile(loss=self.loss_fn, **self.compile_kwargs)
                print(f'Epoch {i}: changing positive prior estimate to {new_prior}')

    def predict(self, X):
        '''
        Get label predictions for a series of examples with the trained model.

        Parameters
        ----------
        X: the examples to get the predictions for
        '''
        return (self.model.predict(X) > 0.5).astype(int)

    def predict_proba(self, X):
        '''
        Get label probabilites for a series of examples with the trained model.

        Parameters
        ----------
        X: the examples to get the probabilities for
        '''
        return self.model.predict(X)

class ProbTagging(PUAlgorithm):
    '''
    Implements the ProbTagging PU-learning algorithm described in "Improving Positive Unlabeled Learning: 
    Practical AUL Estimation and New Training Method for Extremely Imbalanced Data Sets"

    Parameters
    ----------
    knn_num_samples: number of samples to compute the credibility of each sample with KNN
    classifier_class: classifier to use to train each PN dataset
    num_classifiers: number of classifiers to obtain
    classifier_args: arguments to the classifier
    classifier_kwargs: keyword arguments to the classifier
    verbose: print or not debug messages
    '''
    def __init__(self, knn_num_samples, classifier_class, num_classifiers, 
                 classifier_args=[], classifier_kwargs={}, positive_prior=None,
                 random_state=1234, verbose=False):
        
        super().__init__(verbose)

        self.knn_num_samples = knn_num_samples
        self.classifier_class = classifier_class
        self.num_classifiers = num_classifiers
        self.classifier_args = classifier_args
        self.classifier_kwargs = classifier_kwargs
        self.rng = np.random.default_rng(random_state)
        self.positive_prior = positive_prior

    def fit(self, X, y, X_val=None, y_val=None):
        '''
        Fits the ProbTagging model to the given data. Unlike with two-step algorithms, this one does
        not require validation data.

        Samples are first given a credibility score, computed as the proportion of known positives in their
        k-nearest neighbours. Then, "num_classifiers" datasets are created by tagging each example as positive
        or negative with probability given by the credibility score, and a classifier is trained on each dataset.

        Parameters
        ----------
        X: numpy array with shape (num_examples, example_size), with the training features.
        y: numpy array with shape (num_examples), with the labels. They can be 'P' (for positive data),
           or 'U' (for unlabeled data).
        X_val: validation features (unused in this algorithm)
        y_val: validation labels (unused in this algorithm)
        '''
        # Compute credibility scores using k-nearest neighbours
        nbrs = NearestNeighbors(n_neighbors=self.knn_num_samples, n_jobs=-1).fit(X)
        _, indices = nbrs.kneighbors(X)
        unlabeled_samples = (y == 0)
        unlabeled_knn_indices = indices[unlabeled_samples]
        indices_labels = np.vectorize(lambda x: y[x])(unlabeled_knn_indices)
        credibility_scores = np.mean(indices_labels, axis=1)

        # Get "num_classifiers" datasets and train a model on each one of them
        self.classifiers = []
        X_dataset = X[unlabeled_samples]
        for i in range(self.num_classifiers):
            self.print_verbose(f'Training classifier #{i}')
            y_dataset = (self.rng.uniform(size=len(credibility_scores)) < credibility_scores).astype(int)
            self.print_verbose(f'Positives: {np.sum(y_dataset)} // Negatives: {len(y_dataset) - np.sum(y_dataset)}')
            classifier = self.classifier_class(*self.classifier_args, **self.classifier_kwargs)
            classifier.fit(X_dataset, y_dataset)
            self.classifiers.append(classifier)

        if self.positive_prior is None:
            self.positive_prior = np.sum(y == 1) / len(y)


    def predict(self, X):
        '''
        Get label predictions for a series of examples. This model returns the average prediction
        of all trained models

        Parameters
        ----------
        X: the examples to get the predictions for
        '''
        predictions = [classifier.predict(X) for classifier in self.classifiers]
        return (np.mean(predictions, axis=0) > 0.5).astype(int)
    
    def predict_proba(self, X):
        '''
        Get label probabilites for a series of examples. This model returns the average prediction
        of all trained models.

        Parameters
        ----------
        X: the examples to get the probabilities for
        '''
        predictions = [classifier.predict_proba(X)[:,1] for classifier in self.classifiers]
        return np.mean(predictions, axis=0)
    
class TwoStepAlgorithm(PUAlgorithm):
    '''
    Two-step PU algorithms.

    This is the superclass of all two-step positive-unlabeled learning algorithms.
    These consist of a reliable negative detection phase, followed by fitting the
    model itself employed by the algorithm. Optionally, there can be a third, model selection
    step.

    Parameters
    ----------
    negative_detector: algorithm to use for the first step, reliable negative examples detection.

    stop_criterion: method to employ to decide whether to stop or not the iterations of the 
                    second step. Implements the optional, third step.

    random_state: random seed to use for the algorithms that require it.

    verbose: whether to print or not debug messages
    '''

    def __init__(self, negative_detector, stop_criterion, random_state=1234, verbose=False):
        super().__init__(verbose)
        self.negative_detector = negative_detector
        self.stop_criterion = stop_criterion
        self.rng = np.random.default_rng(random_state)

    def fit(self, X, y, X_val, y_val):
        '''
        Fit given data to perform prediction

        This method implements model fitting for two-step PU algorithms, acting as a wrapper for
        the _fit method by previously selecting reliable negatives, and after selecting the positive,
        negative, and unlabeled data.

        Parameters
        ----------
        X: a numpy array with shape (num_examples, example_size), with the training features.

        y: a numpy array with shape (num_examples), with the labels. They can be 'P' (for positive data),
           or 'U' (for unlabeled data).

        X_val: a numpy array with shape (num_validation_examples, example_size), with the validation
                    examples.

        y_val: a numpy array with shape (num_examples), with the validation labels. They can be 'P' (for positive data),
           or 'U' (for unlabeled data).

        Returns
        -------
        self: this same object
        '''
        X_positive = X[y == 1]
        X_unlabeled = X[y == 0]

        negatives, unlabeled = self.negative_detector.detect_negatives(X_positive, X_unlabeled)

        self._fit(X_positive, negatives, unlabeled, X_val, y_val)

        return self

    @abstractmethod
    def _fit(self, positive, negative, unlabeled, X_val, y_val):
        '''
        Internal method for fitting a model to PU-data

        This abstract method is the core of all two-step PU algorithms. It implements the logic
        itself of the algorithm in question, and gets called by fit after it selects reliable
        negatives.

        Parameters
        ----------
        positive: a numpy array with shape (num_positive, example_size), with the positive examples
        
        negative: a numpy array with shape (num_negative, example_size), with the reliable negatives

        unlabeled: a numpy array with shape (num_unlabeled, example_size), with the unlabeled examples

        X_val: a numpy array with shape (num_validation_examples, example_size), with the validation
                    examples.

        y_val: a numpy array with shape (num_examples), with the validation labels. They can be 'P' (for positive data),
           or 'U' (for unlabeled data).
        '''
        pass

class IterativeClassifierAlgorithm(TwoStepAlgorithm):
    '''
    Iterative classifier PU algorithm.

    This algorithm takes a given classifier, and trains it on top of the positive and reliable
    negative examples. Then, a fraction of the most likely negative examples in the unlabeled set
    (as predicted by the classifier) are moved to the negative set. This process is repeated until
    stop criteria is met, or until a maximum number of iterations

    Parameters
    ----------
    negative_detector: algorithm to use for the first step, reliable negative examples detection.

    stop_criterion: method to employ to decide whether to stop or not the iterations of the 
                    second step. Implements the optional, third step.

    classifier_class: Scikit-learn class of the classifier to employ

    frac_to_move: fraction of unlabeled examples to move to the reliable negatives set

    max_iterations_ maximum number of iterations to apply

    classifier_args and classifier_kwargs: arguments to pass to classifier_class constructor

    random_state: random seed to use for the algorithms that require it.

    verbose: whether to print or not debug messages
    '''

    def __init__(self, negative_detector, stop_criterion, classifier_class, 
                 frac_to_move=0.01, max_iterations=100, classifier_args=[], classifier_kwargs={}, 
                 random_state=1234, verbose=False):
        
        super().__init__(negative_detector, stop_criterion, random_state, verbose)

        self.classifier_class = classifier_class
        self.frac_to_move = frac_to_move
        self.max_iterations = max_iterations
        self.classifier_args = classifier_args
        self.classifier_kwargs = classifier_kwargs
        self.validation_results = []

    def _fit(self, positive, negative, unlabeled, X_val, y_val):
        '''
        Internal method for fitting to PU data

        Apply the iterative-classifier two-step PU algorithm, as explained in the docs of
        this class.

        Parameters
        ----------
        positive: a numpy array with shape (num_positive, example_size), with the positive examples
        
        negative: a numpy array with shape (num_negative, example_size), with the reliable negatives

        unlabeled: a numpy array with shape (num_unlabeled, example_size), with the unlabeled examples

        X_val: a numpy array with shape (num_validation_examples, example_size), with the validation
                    examples.

        y_val: a numpy array with shape (num_examples), with the validation labels. They can be 'P' (for positive data),
           or 'U' (for unlabeled data).
        '''
        
        for iter in range(self.max_iterations):
            self.print_verbose(f'Iteration #{iter}')

            classifier = self.classifier_class(*self.classifier_args, **self.classifier_kwargs)

            X_train = np.concatenate([positive, negative])
            y_train = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])

            self.print_verbose(f'Number of negatives: {len(negative)}')
            self.print_verbose(f'Number of positives: {len(positive)}')

            shuffle_idxs = self.rng.permutation(X_train.shape[0])
            X_train = X_train[shuffle_idxs]
            y_train = y_train[shuffle_idxs]
            classifier.fit(X_train, y_train)
            self.print_verbose(f'Finished training classifier')

            should_stop, val_metric = self.stop_criterion.check_stop(classifier, X_val, y_val)
            self.validation_results.append(val_metric)
            self.print_verbose(f'Validation metric for iteration {iter}: {val_metric}')

            if (should_stop):
                break

            else:
                unlabeled_probas = classifier.predict(unlabeled)
                sorted_idxs = np.argsort(unlabeled_probas)
                amount_to_move = int(len(unlabeled) * self.frac_to_move)
                idxs_to_move_negative = sorted_idxs[:amount_to_move]
                idxs_to_move_positive = sorted_idxs[-amount_to_move:]
                negative = np.concatenate([negative, unlabeled[idxs_to_move_negative]], axis=0)
                positive = np.concatenate([positive, unlabeled[idxs_to_move_positive]], axis=0)
                all_idxs_to_remove = np.concatenate([idxs_to_move_negative, idxs_to_move_positive])
                unlabeled = np.delete(unlabeled, all_idxs_to_remove, axis=0)
                self.print_verbose(f'Moving {amount_to_move} from unlabeled to negative and from unlabeled to positive')

        self.classifier = classifier
        return
    
    def predict(self, X):
        '''
        Get label predictions for a series of examples with the trained model.

        Parameters
        ----------
        X: the examples to get the predictions for
        '''
        return self.classifier.predict(X)

    def predict_proba(self, X):
        '''
        Get label probabilites for a series of examples with the trained model.

        Parameters
        ----------
        X: the examples to get the probabilities for
        '''
        return self.classifier.predict_proba(X)[:,1]