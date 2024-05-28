from typing import List, Tuple, Dict
import torch
import numpy as np
from src.data_reader import DataReader
from src.selector.base_selector import BaseSelector
from src.model import speculative_decoder
from src.prompter import Prompter
from tqdm import trange

class AmbigSelector(BaseSelector):
    def __init__(self, model, tokenizer, prompter: Prompter, labels, candidate_num = 10) -> None:

        super(BaseSelector, self).__init__()
        self.prompter = prompter
        self.decoder = speculative_decoder(model, tokenizer)
        self.candidate_num = candidate_num
        self.labels = labels
            
    def select(self, data: DataReader, num: int) -> DataReader:
        selected_idxs = []
        for i in range(len(data)):
            prefix = self.prompter.generate_context([], data[i])
            probs = self.decoder.decode(prefix, self.labels)
            gold_label_idx = self.labels.index(data[i]['label'])
            if gold_label_idx == torch.topk(probs,k=2).indices[1]:
                selected_idxs.append(i)
            if len(selected_idxs) == num:
                break
        return DataReader([data[_] for _ in selected_idxs])