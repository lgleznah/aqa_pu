from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import accuracy_score
from pu.metrics import f1_pu, aul_pu

metrics_dict = {
    'accuracy': accuracy_score,
    'f1': f1_pu,
    'aul': aul_pu
}

class StopCriterion(ABC):
    '''
    Two-step algorithm stop criteria

    This is the superclass of all two-step stop criterion algorithms. These have to implement
    check_stop, to test if the algorithm should stop iterating
    '''

    @abstractmethod
    def check_stop(self, classifier, validation_data, validation_labels):
        '''
        Check if a two-step PU algorithm should stop

        Parameters
        ----------
        classifier: specific classifier to employ for the algorithm.

        validation_data: data upon which to validate the performance of the PU algorithm

        validation_labels: validation data labels. They can be 1 (positive) or 0 (unlabeled)
        '''
        pass

class StopOnMetricDrop(StopCriterion):
    '''
    Stop if a metric drops

    This stop criterion algorithm stops the execution of the two-step algorithm if a given
    metric doesn't improve in an iteration.

    Parameters
    ----------
    metric_name: the name of the metric to track.
    '''

    def __init__(self, metric_name):
        self.metric = metrics_dict[metric_name]
        self.last_metric_value = float('-inf')

    def check_stop(self, classifier, validation_data, validation_labels):
        '''
        Stop execution of the two-step PU algorithm if the given metric doesn't improve.

        Parameters
        ----------
        classifier: specific classifier to employ for the algorithm.

        validation_data: data upon which to validate the performance of the PU algorithm
        '''
        predictions = classifier.predict(validation_data)

        score = self.metric(validation_labels, predictions)
        if (score < self.last_metric_value):
            return True, score
        
        else:
            # Track best metric value
            self.last_metric_value = score
            return False, score
        
class NonStop(StopCriterion):
    '''
    Naïve stop criterion. Never stops, but reports metric results

    Parameters
    ----------
    metric_name: the name of the metric to track.
    '''
    def __init__(self, metric_name):
        self.metric = metrics_dict[metric_name]

    def check_stop(self, classifier, validation_data, validation_labels):
        predictions = classifier.predict(validation_data)
        score = self.metric(validation_labels, predictions)
        return False, score