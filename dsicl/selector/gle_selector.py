from typing import List, Tuple, Dict
import torch
import numpy as np
from dsicl.data_reader import DataReader
from dsicl.selector.base_selector import BaseSelector
from dsicl.model import speculative_decoder
from dsicl.prompter import Prompter
from tqdm import trange

class GLESelector(BaseSelector):
    def __init__(self, model, tokenizer, prompter: Prompter, labels, candidate_num = 10) -> None:

        super(BaseSelector, self).__init__()
        self.prompter = prompter
        self.decoder = speculative_decoder(model, tokenizer)
        self.candidate_num = candidate_num
        self.labels = labels
            
    def entropy(self, probs):
        return torch.sum(-probs*torch.log2(probs+1e-5))

    def select(self, data: DataReader, shots: int, val_data: DataReader) -> DataReader:
        
        train_data = data
        
        candidate_indexs = [np.random.choice(len(train_data), shots,replace=False) for _ in range(self.candidate_num)]

        global_preds = torch.zeros((self.candidate_num, len(self.labels))).to(self.decoder.device)
        
        for i in trange(len(candidate_indexs)):
            demos = [train_data[_] for _ in candidate_indexs[i]]
            for j in range(len(val_data)):
                prefix = self.prompter.generate_context(demos, val_data[j])
                probs = self.decoder.decode(prefix, self.labels)
                global_preds[i][probs.argmax()] += 1
            global_preds[i] /= torch.sum(global_preds[i])

        gle = torch.tensor([self.entropy(global_preds[_]) for _ in range(global_preds.size()[0])])
        best_index = candidate_indexs[torch.argmax(gle)]

        return DataReader([train_data[_] for _ in best_index])