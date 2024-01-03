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
    '''
    Parent class for all PU and PN dataset loader classes.
    '''
    @abstractmethod
    def load_data(self):
        '''
        Abstract method for loading a dataset. Return values depend on the specific kind dataset
        being loaded.
        '''
        pass


class CSVLoader(Loader):
    '''
    This class loads a full CSV, returning the paths of the images contained therein.

    Parameters
    ----------
    file: CSV file with the paths
    path_col: name of the CSV column with the paths
    img_root: root folder containing all the images of this dataset
    '''
    def __init__(self, file, path_col, img_root):
        self.file = file
        self.img_root = img_root
        self.path_col = path_col

    def load_data(self):
        df = pd.read_csv(self.file)

        paths = df[[self.path_col]].squeeze().apply(lambda path: os.path.join(self.img_root, path)).to_numpy()
        return paths
    

class FullCSVLoader(Loader):
    def __init__(self, file, path_col, img_root):
        self.file = file
        self.img_root = img_root
        self.path_col = path_col

    def load_data(self):
        df = pd.read_csv(self.file)

        df[self.path_col] = df[self.path_col].squeeze().apply(lambda path: os.path.join(self.img_root, path))
        return df

class SingleCSVLoader(Loader):
    '''
    This class loads a full CSV, with both positive and unlabeled examples. Examples are classified
    according to a function.

    Parameters
    ----------
    file: CSV file with the paths
    path_col: name of the CSV column with the paths
    img_root: root folder containing all the images of this dataset
    positive_fn: bool function that determines if an example is positive (True) or unlabeled (False)
    '''
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
    '''
    This class loads a full CSV, with both positive and unlabeled examples. Examples are classified
    according to a reliable positive function. Additionaly, a PN test set is returned, in which positives
    and negatives are classified according to a positive function

    Parameters
    ----------
    file: CSV file with the paths
    path_col: name of the CSV column with the paths
    img_root: root folder containing all the images of this dataset
    reliable_positive_fn: bool function that determines if an example is positive (True) or unlabeled (False)
    positive_fn: bool function that determines if an example is positive (True) or negative (False)
    test_frac: fraction of data to keep as test set
    random_state: random seed for the RNG
    '''
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
        df_positive = df[df.apply(lambda x: self.positive_fn(x, df), axis=1)]
        df_negative = df[df.apply(negate(lambda x: self.positive_fn(x, df)), axis=1)]

        test_positive_idxs = self.rng.choice(len(df_positive), size=int(self.test_frac * len(df_positive)), replace=False)
        test_negative_idxs = self.rng.choice(len(df_negative), size=int(self.test_frac * len(df_negative)), replace=False)

        df_test_positive = df_positive.iloc[test_positive_idxs]
        df_test_negative = df_negative.iloc[test_negative_idxs]

        df_positive = df_positive.reset_index(drop=True)
        df_negative = df_negative.reset_index(drop=True)
        df_positive = df_positive.drop(test_positive_idxs)
        df_negative = df_negative.drop(test_negative_idxs)
        df = pd.concat([df_positive, df_negative])

        df_train_positive = df[df.apply(lambda x: self.reliable_positive_fn(x, df), axis=1)]
        df_train_unlabeled = df[df.apply(negate(lambda x: self.reliable_positive_fn(x, df)), axis=1)]

        paths_train_positive = df_train_positive[[self.path_col]].squeeze().apply(lambda path: os.path.join(self.img_root, path)).to_numpy()
        paths_train_unlabeled = df_train_unlabeled[[self.path_col]].squeeze().apply(lambda path: os.path.join(self.img_root, path)).to_numpy()
        paths_test_positive = df_test_positive[[self.path_col]].squeeze().apply(lambda path: os.path.join(self.img_root, path)).to_numpy()
        paths_test_negative = df_test_negative[[self.path_col]].squeeze().apply(lambda path: os.path.join(self.img_root, path)).to_numpy()

        return paths_train_positive, paths_train_unlabeled, paths_test_positive, paths_test_negative
        
class SinglePNCSVWithTestLoader(Loader):
    '''
    This class loads a full CSV, with both positive and negative examples. Examples are classified
    according to a positive function. A test set is also returned.

    Parameters
    ----------
    file: CSV file with the paths
    path_col: name of the CSV column with the paths
    img_root: root folder containing all the images of this dataset
    positive_fn: bool function that determines if an example is positive (True) or negative (False)
    test_frac: fraction of data to keep as test set
    random_state: random seed for the RNG
    '''
    def __init__(self, file, path_col, img_root, positive_fn, test_frac, random_state):
        self.file = file
        self.img_root = img_root
        self.path_col = path_col
        self.positive_fn = positive_fn
        self.test_frac = test_frac

        self.rng = np.random.default_rng(seed=random_state)

    def load_data(self):
        df = pd.read_csv(self.file)
        df_positive = df[df.apply(self.positive_fn, axis=1)]
        df_negative = df[df.apply(negate(self.positive_fn), axis=1)]

        test_positive_idxs = self.rng.choice(len(df_positive), size=int(self.test_frac * len(df_positive)), replace=False)
        test_negative_idxs = self.rng.choice(len(df_negative), size=int(self.test_frac * len(df_negative)), replace=False)

        df_test_positive = df_positive.iloc[test_positive_idxs]
        df_test_negative = df_negative.iloc[test_negative_idxs]

        df_positive = df_positive.reset_index(drop=True)
        df_negative = df_negative.reset_index(drop=True)
        df_positive = df_positive.drop(test_positive_idxs)
        df_negative = df_negative.drop(test_negative_idxs)

        paths_train_positive = df_positive[[self.path_col]].squeeze().apply(lambda path: os.path.join(self.img_root, path)).to_numpy()
        paths_train_negative = df_negative[[self.path_col]].squeeze().apply(lambda path: os.path.join(self.img_root, path)).to_numpy()
        paths_test_positive = df_test_positive[[self.path_col]].squeeze().apply(lambda path: os.path.join(self.img_root, path)).to_numpy()
        paths_test_negative = df_test_negative[[self.path_col]].squeeze().apply(lambda path: os.path.join(self.img_root, path)).to_numpy()

        return paths_train_positive, paths_train_negative, paths_test_positive, paths_test_negative