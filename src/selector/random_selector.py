from typing import List, Tuple, Dict
import torch
import numpy as np
from src.data_reader import DataReader
from src.selector.base_selector import BaseSelector

class RandomSelector(BaseSelector):
    def __init__(self) -> None:
        pass

    def select(self, data: DataReader, num ,val_data: DataReader = None, seed = None) -> DataReader:
        if seed is not None:
            np.random.seed(seed)
        idxs = np.random.choice(len(data), num, replace=False)
        return DataReader([data[_] for _ in idxs])