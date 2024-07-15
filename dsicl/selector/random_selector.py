from typing import List, Tuple, Dict
import torch
import numpy as np
from dsicl.data_reader import DataReader
from dsicl.selector.base_selector import BaseSelector

class RandomSelector(BaseSelector):
    def __init__(self) -> None:
        pass

    def select(self, data: DataReader, num) -> DataReader:
        idxs = np.random.choice(len(data), num, replace=False)
        return DataReader([data[_] for _ in idxs])