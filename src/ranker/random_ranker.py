import numpy as np
import torch
from typing import List, Tuple, Dict
from src.data_reader import DataReader
from src.ranker.base_ranker import BaseRanker

class RandomRanker(BaseRanker):
    def __init__(self) -> None:
        super(BaseRanker, self).__init__()

    def rank(self, demos: DataReader, sample: Dict = None, num = None) -> DataReader:
        if num is None:
            idxs = np.random.choice(len(demos), len(demos), replace=False)
        else:
            idxs = np.random.choice(len(demos), num, replace=False)
        return DataReader([demos[_] for _ in idxs])
    
    def batch_rank(self, demos_l: List[DataReader] , data: DataReader, num = None) -> List[DataReader]:
        return [self.rank(demos=demos_l[_], sample=data[_], num=num) for _ in range(len(data))]
