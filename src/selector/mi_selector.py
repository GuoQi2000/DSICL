from typing import List, Tuple, Dict
import torch
import numpy as np
from src.data_reader import DataReader
from src.selector.base_selector import BaseSelector
from src.model import speculative_decoder
from src.prompter import Prompter
from tqdm import trange

class MISelector(BaseSelector):
    def __init__(self, model, tokenizer, prompter: Prompter, labels, candidate_num = 10) -> None:

        super(BaseSelector, self).__init__()
        self.prompter = prompter
        self.decoder = speculative_decoder(model, tokenizer)
        self.candidate_num = candidate_num
        self.labels = labels
            
    def entropy(self, probs):
        return torch.sum(-probs*torch.log2(probs+1e-5))

    def select(self, data: DataReader, num: int) -> DataReader:

        val_data, train_data = data.split(num=100)

        candidate_indexs = [np.random.choice(len(train_data), num,replace=False) for _ in range(self.candidate_num)]

        avg_probs = torch.zeros((self.candidate_num, len(self.labels))).to(self.decoder.device)
        avg_entropy = torch.zeros(self.candidate_num).to(self.decoder.device)
        
        for i in trange(len(candidate_indexs)):
            demos = [train_data[_] for _ in candidate_indexs[i]]
            for j in range(len(val_data)):
                prefix = self.prompter.generate_context(demos, val_data[j])
                probs = self.decoder.decode(prefix, self.labels)
                avg_probs[i] += probs
                avg_entropy[i] += self.entropy(probs)
            avg_probs[i] /= len(val_data)
            avg_entropy[i] /= len(val_data)

        mi_values = torch.tensor([self.entropy(avg_probs[_]) - avg_entropy[_] for _ in range(len(candidate_indexs))])
        
        best_index = candidate_indexs[torch.argmax(mi_values)]

        return DataReader([train_data[_] for _ in best_index])