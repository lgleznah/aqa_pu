from abc import ABC, abstractmethod

import pandas as pd
import os

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