from abc import ABC, abstractmethod

import numpy as np

class PUAlgorithm(ABC):
    '''
    Learn from positive and unlabeled data.

    This is the superclass for all algorithms that learn from positive
    and unlabeled data.

    The structure of these algorithms is mostly based on the survey
    "Learning From Positive and Unlabeled Data: A Survey", by Jessa Bekker and Jesse Davis.
    '''

    @abstractmethod
    def fit(self, X, y, validation):
        '''
        Fit given data to perform prediction.

        This abstract method is the core of all PU algorithms, and it must be implemented by all
        subclasses of it. It takes positive and unlabeled data, and trains a model on top of it.
        
        Parameters
        ----------
        X: a numpy array with shape (num_examples, example_size), with the training features.

        y: a numpy array with shape (num_examples), with the labels. They can be 'P' (for positive data),
           or 'U' (for unlabeled data).

        validation: a numpy array with shape (num_validation_examples, example_size), with the validation
                    examples. All these are assumed to be positive, and are used to give estimates of
                    accuracy or F1 scores.

        Returns
        -------
        self: this same object
        '''
        pass

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
    '''

    def __init__(self, negative_detector, stop_criterion, random_state=1234):
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
        X_positive = X[y == 'P']
        X_unlabeled = X[y == 'U']

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
    '''

    def __init__(self, negative_detector, stop_criterion, classifier_class, frac_to_move=0.01, max_iterations=100, classifier_args=[], classifier_kwargs={}, random_state=1234):
        super().__init__(negative_detector, stop_criterion, random_state)

        self.classifier_class = classifier_class
        self.frac_to_move = frac_to_move
        self.max_iterations = max_iterations
        self.classifier_args = classifier_args
        self.classifier_kwargs = classifier_kwargs
        self.validation_results = []

    def _fit(self, positive, negative, unlabeled, validation):
        '''
        Internal method for fitting to PU data

        Apply the iterative-classifier two-step PU algorithm, as explained in the docs of
        this class.

        Parameters
        ----------
        positive: a numpy array with shape (num_positive, example_size), with the positive examples
        
        negative: a numpy array with shape (num_negative, example_size), with the reliable negatives

        unlabeled: a numpy array with shape (num_unlabeled, example_size), with the unlabeled examples

        validation: a numpy array with shape (num_validation_examples, example_size), with the
                    validation examples
        '''
        
        for _ in range(self.max_iterations):
            classifier = self.classifier_class(self.classifier_args, self.classifier_kwargs)

            X_train = np.concatenate(positive, negative)
            y_train = np.concatenate(np.ones(len(positive)), np.zeros(len(negative)))

            shuffle_idxs = self.rng.permutation(X_train.shape[0])
            X_train = X_train[shuffle_idxs]
            y_train = y_train[shuffle_idxs]
            classifier.fit(X_train, y_train)

            should_stop, val_metric = self.stop_criterion(classifier, validation)
            self.validation_results.append(val_metric)
            if (should_stop):
                self.classifier = classifier
                break

            else:
                unlabeled_probas = classifier.predict_proba(unlabeled)
                sorted_idxs = np.argsort(unlabeled_probas)
                amount_to_move = len(unlabeled) * self.frac_to_move
                idxs_to_move = sorted_idxs[:amount_to_move]
                negative = np.concatenate([negative, unlabeled[idxs_to_move]], axis=0)
                unlabeled = np.delete(unlabeled, idxs_to_move, axis=0)

        return