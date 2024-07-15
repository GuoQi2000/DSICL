import numpy as np
import torch
from typing import List, Tuple, Dict
from dsicl.data_reader import DataReader
from dsicl.ranker.base_ranker import BaseRanker

class RandomRanker(BaseRanker):
    def __init__(self) -> None:
        super(BaseRanker, self).__init__()

    def rank(self, demos: DataReader, shots: int, sample: Dict = None,) -> DataReader:
        
        idxs = np.random.choice(len(demos), shots, replace=False)
        return DataReader([demos[_] for _ in idxs])
    
    def batch_rank(self, demos_l: List[DataReader] , data: DataReader, shots: int) -> List[DataReader]:
        return [self.rank(demos=demos_l[_], sample=data[_], shots=shots) for _ in range(len(data))]
