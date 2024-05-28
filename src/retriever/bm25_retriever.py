import numpy as np
import os
from src.model import get_encoder
from tqdm import trange, tqdm
import torch
from typing import List, Tuple, Dict, Union
from src.retriever.base_retriever import BaseRetriever
from src.data_reader import DataReader
from src.prompter import Prompter
from rank_bm25 import BM25Okapi

class BM25Retriever(BaseRetriever):

    def __init__(self, 
                 tokenizer,
                 prompter: Prompter,
                 data: DataReader = None) -> None:
        
        super(BaseRetriever, self).__init__()

        self.tokenizer = tokenizer
        self.prompter = prompter
        self.data = data

        self.tokenized_corpus = [tokenizer.tokenize(self.prompter.generate_prompt(doc)) for doc in self.data]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, sample: Dict, k: int) -> DataReader:
        q = self.tokenizer.tokenize(self.prompter.generate_prompt(sample))
        scores = self.bm25.get_scores(q)
        scores = torch.tensor(scores)

        idxs = scores.argsort()[-k:]

        return DataReader([self.data[_] for _ in idxs])
    

    def batch_retrieve(self, data: DataReader, k: int) -> List[DataReader]:

        return [self.retrieve(d,k) for d in tqdm(data)]
    