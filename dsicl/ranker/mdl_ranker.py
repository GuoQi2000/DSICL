import numpy as np
import torch
from typing import List, Tuple, Dict
from dsicl.data_reader import DataReader
from dsicl.ranker.base_ranker import BaseRanker
from dsicl.model import speculative_decoder
from dsicl.prompter import Prompter
from tqdm import trange

class MDLRanker(BaseRanker):
    def __init__(self, model, tokenizer, prompter: Prompter, labels, candidate_num = 10) -> None:
        super(BaseRanker, self).__init__()
        
        self.prompter = prompter
        self.decoder = speculative_decoder(model, tokenizer)
        self.candidate_num = candidate_num
        self.labels = labels

    def mdl(self, probs):
        return torch.sum(-probs*torch.log2(probs))

    def rank(self, demos: DataReader, sample: Dict, shots: int) -> DataReader:
        
        candidate_orders = [np.random.choice(len(demos), shots, replace=False) for _ in range(self.candidate_num)]
        mdl_values = []
        for order in candidate_orders:
            prefix = self.prompter.generate_context([demos[_] for _ in order],sample)
            probs = self.decoder.decode(prefix, self.labels)
            mdl_values.append(self.mdl(probs))

        best_order = candidate_orders[torch.argmin(torch.tensor(mdl_values))]
        return DataReader([demos[_] for _ in best_order])

    def batch_rank(self, demos_l: List[DataReader] , data: DataReader, shots: int) -> List[DataReader]:
        return [self.rank(demos=demos_l[_], sample=data[_], shots=shots) for _ in trange(len(data))]