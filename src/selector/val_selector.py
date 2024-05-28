from typing import List, Tuple, Dict
import torch
from torch import Tensor
import numpy as np
from src.data_reader import DataReader
from src.selector.base_selector import BaseSelector
from src.model import speculative_decoder
from src.prompter import Prompter
from tqdm import trange, tqdm

class ValSelector(BaseSelector):
    def __init__(self, model, tokenizer, prompter: Prompter, labels) -> None:

        super(BaseSelector, self).__init__()
        self.prompter = prompter
        self.decoder = speculative_decoder(model, tokenizer)
        self.labels = labels

    def compute_acc(self, demos: DataReader, data: DataReader):
        acc = 0
        for i in range(len(data)):
            gold_label_idx = self.labels.index(data[i]['label'])
            prefix = self.prompter.generate_context(demos, data[i])
            probs = self.decoder.decode(prefix, self.labels)
            if probs.argmax() == gold_label_idx:
                acc += 1
        return acc / len(data)
            
    def select(self, data: DataReader, num: int, val_size: int = 100, candidate_num = 100) -> DataReader:
        val_data, candidates = data.split(num=val_size, random=True)
        accs = []
        demos_l = []
        for i in trange(candidate_num):
            demos = candidates.split(num=num)[0]
            demos_l.append(demos)
            accs.append(self.compute_acc(demos, val_data))
        accs = torch.tensor(accs)
        return demos_l[accs.argmax()]