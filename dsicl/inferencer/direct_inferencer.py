import numpy as np
import torch
from typing import List, Tuple, Dict
from tqdm import trange
from dsicl.data_reader import DataReader
from dsicl.prompter import Prompter
from dsicl.inferencer.base_inferencer import BaseInferencer
from dsicl.model import speculative_decoder

class DirectInferencer(BaseInferencer):

    def __init__(self, model, tokenizer, prompter: Prompter, labels) -> None:
        super(BaseInferencer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.decoder = speculative_decoder(self.model, self.tokenizer)
        self.prompter = prompter
        self.labels = labels

    def infer(self, demos: DataReader, sample: Dict):
        context = self.prompter.generate_context(demos, sample)
        # print(context)
        label_idx = self.decoder.decode(context, self.labels).argmax()
        return self.labels[label_idx]
    
    def batch_infer(self, demos_l: List[DataReader] , data: DataReader) -> List[str]:
        return [self.infer(demos_l[_], data[_]) for _ in trange(len(data))]


        