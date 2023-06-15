from abc import ABC, abstractmethod

class PUAlgorithm(ABC):

    @abstractmethod
    def fit(X, y):
        pass

