from typing import List, Tuple, Dict
import torch
from torch import Tensor
import numpy as np
from src.data_reader import DataReader
from src.selector.base_selector import BaseSelector
from src.model import speculative_decoder
from src.prompter import Prompter
from tqdm import trange, tqdm

class DifferenceSelector(BaseSelector):
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
        

    def compute_difference(self, sample: Dict, data: DataReader):
        difference = 0
        score = 0
        fake_sample = sample.copy()
        if sample['label'] == self.labels[0]:
            fake_sample['label'] = self.labels[1]
        else:
            fake_sample['label'] = self.labels[0]
        for i in range(len(data)):
            gold_label_idx = self.labels.index(data[i]['label'])
            prefix = self.prompter.generate_context([sample], data[i])
            fake_prefix = self.prompter.generate_context([fake_sample], data[i])
            probs = self.decoder.decode(prefix, self.labels)
            fake_probs = self.decoder.decode(fake_prefix, self.labels)
            if not probs.argmax() == fake_probs.argmax():
                difference += 1
            if probs.argmax() == gold_label_idx:
                score += 1
        return difference / len(data), score / len(data)
            
    def select(self, data: DataReader, num: int, val_size: int = 100, candidate_num = 100, val_data:DataReader = None) -> DataReader:
        if val_data is None:
            val_data, others = data.split(num=val_size, random=True)
            candidates = others.split(num=candidate_num, random=True)[0]
        else:
            candidates = data
        differences = []
        scores = []
        for sample in tqdm(candidates):
            d, s = self.compute_difference(sample, val_data)
            differences.append(d)
            scores.append(s)
        differences = torch.tensor(differences)
        scores = torch.tensor(scores)
        # idxs = torch.argsort(-(1/(1/scores + 1/differences)))
        idxs = torch.argsort((differences))
        best_idxs = []
       # best_idxs = torch.topk(differences, k=num).indices
        l = int(num / len(self.labels))
        for i in range(len(self.labels)):
            c = 0
            for idx in idxs:
                if candidates[idx]['label'] == self.labels[i]:
                    best_idxs.append(idx)
                    c+=1
                if c == l:
                    break
        best_demos = [candidates[_] for _ in best_idxs]
        np.random.shuffle(best_demos)
        return DataReader(best_demos)