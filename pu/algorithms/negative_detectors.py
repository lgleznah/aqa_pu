from abc import ABC, abstractmethod

import numpy as np

class NegativeDetector(ABC):
    '''
    Reliable negative detector in PU data

    This is the parent class of all the reliable negative examples detector
    for positive-unlabeled problems
    '''

    @abstractmethod
    def detect_negatives(self, positives, unlabeled):
        '''
        Detect reliable negatives in unlabeled data

        This is the abstract function which must be overriden by children of NegativeDetector
        to implement all the different negative detectors.

        Children of this function should return a tuple of two values: the first one is the
        list of reliable negative examples, and the second one is the list of remaining unlabeled data
        '''
        pass

class NaiveDetector(NegativeDetector):
    '''
    Naive negative detector

    Just returns a random fraction of the unlabeled data as negative data

    Parameters
    ----------
    random_state: random seed to use for the algorithms that require it.

    frac: fraction of unlabeled data to return as "reliable" negatives

    Returns
    -------
    (negative, remaining): a tuple of two values, containing the reliable negatives
                           and the remaining examples, respectively.
    '''

    def __init__(self, frac, random_state):
        super().__init__()
        self.rng = np.random.default_rng(random_state)
        self.frac = frac

    def detect_negatives(self, positives, unlabeled):
        amount = int(len(unlabeled) * self.frac)
        idxs = self.rng.permutation(len(unlabeled))[:amount]

        return unlabeled[idxs], np.delete(unlabeled, idxs, axis=0)