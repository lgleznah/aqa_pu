from abc import ABC, abstractmethod
from functools import wraps

import pandas as pd
import os
import numpy as np

def negate(f):
    @wraps(f)
    def g(*args,**kwargs):
        return not f(*args,**kwargs)
    g.__name__ = f'negate({f.__name__})'
    return g

class Loader(ABC):

    @abstractmethod
    def load_data(self):
        pass


class CSVLoader(Loader):

    def __init__(self, file, path_col, img_root):
        self.file = file
        self.img_root = img_root
        self.path_col = path_col

    def load_data(self):
        df = pd.read_csv(self.file)

        paths = df[[self.path_col]].squeeze().apply(lambda path: os.path.join(self.img_root, path)).to_numpy()
        return paths
    
class SingleCSVLoader(Loader):

    def __init__(self, file, path_col, img_root, positive_fn):
        self.file = file
        self.img_root = img_root
        self.path_col = path_col
        self.positive_fn = positive_fn

    def load_data(self):
        df = pd.read_csv(self.file)
        df_positive = df.apply(self.positive_fn, axis=1)
        df_unlabeled = df.apply(negate(self.positive_fn), axis=1)

        paths_positive = df_positive[[self.path_col]].squeeze().apply(lambda path: os.path.join(self.img_root, path)).to_numpy()
        paths_unlabeled = df_unlabeled[[self.path_col]].squeeze().apply(lambda path: os.path.join(self.img_root, path)).to_numpy()

        return paths_positive, paths_unlabeled
    
class SingleCSVWithTestLoader(Loader):

    def __init__(self, file, path_col, img_root, reliable_positive_fn, positive_fn, test_frac, random_state):
        self.file = file
        self.img_root = img_root
        self.path_col = path_col
        self.reliable_positive_fn = reliable_positive_fn
        self.positive_fn = positive_fn
        self.test_frac = test_frac

        self.rng = np.random.default_rng(seed=random_state)

    def load_data(self):
        df = pd.read_csv(self.file)
        df_positive = df.apply(self.positive_fn, axis=1)
        df_negative = df.apply(negate(self.positive_fn), axis=1)

        test_positive_idxs = self.rng.choice(len(df_positive), size=self.test_frac * len(df_positive), replace=False)
        test_negative_idxs = self.rng.choice(len(df_negative), size=self.test_frac * len(df_negative), replace=False)

        df_test_positive = df_positive.iloc[test_positive_idxs]
        df_test_negative = df_negative.iloc[test_negative_idxs]

        df_positive = df_positive[~df_positive.index.isin(test_positive_idxs)]
        df_negative = df_negative[~df_negative.index.isin(test_negative_idxs)]
        df = pd.concat([df_positive, df_negative])

        df_train_positive = df.apply(self.reliable_positive_fn, axis=1)
        df_train_unlabeled = df.apply(negate(self.reliable_positive_fn), axis=1)

        paths_train_positive = df_train_positive[[self.path_col]].squeeze().apply(lambda path: os.path.join(self.img_root, path)).to_numpy()
        paths_train_unlabeled = df_train_unlabeled[[self.path_col]].squeeze().apply(lambda path: os.path.join(self.img_root, path)).to_numpy()
        paths_test_positive = df_test_positive[[self.path_col]].squeeze().apply(lambda path: os.path.join(self.img_root, path)).to_numpy()
        paths_test_negative = df_test_negative[[self.path_col]].squeeze().apply(lambda path: os.path.join(self.img_root, path)).to_numpy()

        return paths_train_positive, paths_train_unlabeled, paths_test_positive, paths_test_negative
        
