import numpy as np
import torch
from typing import List, Tuple, Dict
from tqdm import trange
from src.data_reader import DataReader
from src.prompter import Prompter
from src.inferencer.base_inferencer import BaseInferencer
from src.model import speculative_decoder

class DirectInferencer(BaseInferencer):

    def __init__(self, model, tokenizer, prompter: Prompter, labels) -> None:
        super(BaseInferencer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.decoder = speculative_decoder(self.model, self.tokenizer)
        self.prompter = prompter
        self.labels = labels

    def infer(self, demos: DataReader, sample: Dict):#并行解码，目标标签并行执行model推理（batch size = 标签num），显存占用大，以推理空间（显存）换推理时间，推理速度变快
        context = self.prompter.generate_context(demos, sample)
        label_idx = self.decoder.decode(context, self.labels).argmax()
        return self.labels[label_idx]
    
    def batch_infer(self, demos_l: List[DataReader] , data: DataReader) -> List[str]:
        return [self.infer(demos_l[_], data[_]) for _ in trange(len(data))]


        