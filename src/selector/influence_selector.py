from typing import List, Tuple, Dict
import torch
import json
from torch import Tensor
import numpy as np
from src.data_reader import DataReader
from src.selector.base_selector import BaseSelector
from src.model import speculative_decoder
from src.prompter import Prompter
from tqdm import trange, tqdm

class InfluenceSelector(BaseSelector):
    def __init__(self, model, tokenizer, prompter: Prompter, labels) -> None:

        super(BaseSelector, self).__init__()
        self.prompter = prompter
        self.decoder = speculative_decoder(model, tokenizer)
        self.labels = labels

    def subset_collection(self, data: DataReader, subset_num: int, subset_size: int) -> List[DataReader]:
        return [data.split(num=subset_size)[0] for _ in range(subset_num)]
        
    def compute_subset_score(self, demos: DataReader, val_data: DataReader, metric='acc') -> Tensor:
        score = 0
        if metric == 'acc':
            for i in range(len(val_data)):
                gold_label_idx = self.labels.index(val_data[i]['label'])
                prefix = self.prompter.generate_context(demos, val_data[i])
                probs = self.decoder.decode(prefix, self.labels)
                score += probs.argmax() == gold_label_idx
        return torch.tensor(score / len(val_data))

    def compute_influence(self, subsets: List[DataReader], scores: Tensor) -> Tuple[DataReader, Tensor]:
        candidates = []
        for subset in subsets:
            for sample in subset:
                if not sample in candidates:
                    candidates.append(sample)
        influences = []
        for sample in candidates:
            including = [0, 0]
            omitting = [0, 0]
            for subset, score in zip(subsets, scores):
                if sample in subset:
                    including[0] += score
                    including[1] += 1
                else:
                    omitting[0] += score
                    omitting[1] += 1
            including_score = including[0] / including[1]
            if omitting[1] == 0:
                omitting_score = 1
            else:
                omitting_score = omitting[0] / omitting[1]
            influences.append(including_score - omitting_score)
        return DataReader(candidates), torch.tensor(influences)

          
    def select(self, data: DataReader, num: int, subset_num = 1000, val_size = 200) -> DataReader:
        val_data, train_data = data.split(num=val_size)
        train_data = train_data.split(num=200)[0]
        subsets = self.subset_collection(train_data, subset_num, num)
        scores = torch.tensor([self.compute_subset_score(subset, val_data) for subset in tqdm(subsets)])
        candidates, influences = self.compute_influence(subsets, scores)
        best_idxs = torch.topk(influences, num).indices
        
        return DataReader([candidates[_] for _ in best_idxs])