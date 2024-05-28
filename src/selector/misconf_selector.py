from typing import List, Tuple, Dict
import torch
from torch import Tensor
import numpy as np
from src.data_reader import DataReader
from src.selector.base_selector import BaseSelector
from src.model import speculative_decoder
from src.prompter import Prompter
from tqdm import trange

class MisconfSelector(BaseSelector):
    def __init__(self, model, tokenizer, prompter: Prompter, labels) -> None:

        super(BaseSelector, self).__init__()
        self.prompter = prompter
        self.decoder = speculative_decoder(model, tokenizer)
        self.labels = labels

    def reflect(self, mis_confs: Tensor, demos: DataReader, candidates: DataReader, replace_num: int) -> Tuple[DataReader, DataReader]:
        top_mis_conf_idxs = torch.topk(mis_confs, replace_num).indices
        new_demos = DataReader([candidates[_] for _ in top_mis_conf_idxs]+demos[replace_num:])
        new_candidates = DataReader([candidates[_] for _ in range(len(candidates)) if not _ in top_mis_conf_idxs]+demos[:replace_num])
        return new_demos, new_candidates
        

    def compute_misconfidence(self, demos: DataReader, data: DataReader):
        mis_conf = []
        for i in trange(len(data)):
            gold_label_idx = self.labels.index(data[i]['label'])
            prefix = self.prompter.generate_context(demos, data[i])
            probs = self.decoder.decode(prefix, self.labels)
            if probs.argmax() == gold_label_idx:
                mis_conf.append(torch.topk(probs,2).values[1] / (probs[gold_label_idx]+1e-3))
            else:
                mis_conf.append(probs.argmax() / (probs[gold_label_idx]+1e-3)) 
        return torch.tensor(mis_conf)
            
    def select(self, data: DataReader, num: int, replace_num:int, iters = 1) -> DataReader:
        initial_demos, candidates = data.split(num=num)
        for i in range(iters):
            mis_confs = self.compute_misconfidence(demos=initial_demos, data=candidates)
            initial_demos, candidates = self.reflect(mis_confs, initial_demos, candidates, replace_num)

        return initial_demos