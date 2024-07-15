from typing import List, Tuple, Dict
import torch
import numpy as np
from dsicl.data_reader import DataReader
from dsicl.selector.base_selector import BaseSelector
from dsicl.model import speculative_decoder
from dsicl.prompter import Prompter
from tqdm import trange

class FairnessSelector(BaseSelector):
    def __init__(self, model, tokenizer, prompter: Prompter, labels, candidate_num = 100) -> None:

        super(BaseSelector, self).__init__()
        self.prompter = prompter
        self.decoder = speculative_decoder(model, tokenizer)
        self.candidate_num = candidate_num
        self.labels = labels
            
    def entropy(self, probs):
        return torch.sum(-probs*torch.log2(probs+1e-5))

    def select(self, data: DataReader, shots: int) -> DataReader:
        train_data = data
        candidate_indexs = np.random.choice(len(train_data), min(self.candidate_num, len(train_data)), replace=False)
        fairness_l = []
        content_free_sample = {}
        for k,v in train_data[0].items():
            if k == 'label':
                content_free_sample[k] = v
            else:
                content_free_sample[k] = '[N/A]'
        for i in trange(len(candidate_indexs)):
            demos = [train_data[candidate_indexs[i]]]
            prefix = self.prompter.generate_context(demos, content_free_sample)
            probs = self.decoder.decode(prefix, self.labels)
            fairness = self.entropy(probs)
            fairness_l.append(fairness)

        best_index = candidate_indexs[torch.argsort(torch.tensor(fairness_l))[-shots:]]

        return DataReader([train_data[_] for _ in best_index])