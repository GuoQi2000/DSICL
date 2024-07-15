import numpy as np
import os
from dsicl.model import get_encoder
from tqdm import trange, tqdm
import torch
from typing import List, Tuple, Dict, Union
from dsicl.retriever.base_retriever import BaseRetriever
from dsicl.data_reader import DataReader
from dsicl.prompter import Prompter

class PromptingRetriever(BaseRetriever):

    def __init__(self, 
                 model,
                 tokenizer,
                 prompter: Prompter,
                 demo_size: int,
                 data: DataReader = None,
                 embedding_path: str = None) -> None:
        
        super(BaseRetriever, self).__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.demo_size = demo_size

        if not data is None:
            self.data = data
            self.embeddings = None

        if not embedding_path is None:
            self.embeddings = torch.load(embedding_path)
        
        self.demos, self.anchors = self.data.split(num=self.demo_size, random=True)

    def encode(self, x: str) -> torch.FloatTensor:
        with torch.no_grad():
            inputs = self.tokenizer(x,max_length=1000, truncation=True, padding= False, return_tensors="pt").to(self.model.device)
            logits = self.model(**inputs).logits[0, -1,:]
            return logits

    def save_embeddings(self, path):
        torch.save(self.embeddings, path)

    def build_embeddings(self) -> None:
        embeddings = []
        for i in trange(len(self.anchors)):
            x = self.prompter.generate_context(self.demos, self.anchors[i])
            v = self.encode(x)
            embeddings.append(v)
        self.embeddings = torch.stack(embeddings)
#        torch.save(embeddings, f"embedding/{self.paras['task']}_{self.paras['encoder']}.pt")

    def retrieve(self, sample: Dict, shots: int) -> DataReader:
        q = self.encode(self.prompter.generate_context(self.demos, sample))
        distances = torch.sum((q-self.embeddings)**2, dim=1)**0.5
        idxs = distances.argsort()[:shots]
        return DataReader([self.data[_] for _ in idxs][::-1])
    
    def batch_retrieve(self, data: DataReader, shots: int) -> List[DataReader]:

        return [self.retrieve(d, shots) for d in tqdm(data)]
        # q = self.encode(self.prompter.generate_prompt(sample))
        # distances = torch.sum((q-self.embeddings)**2, dim=1)**0.5
        # idxs = distances.argsort()[:k]
        # return DataReader([self.data[_] for _ in idxs][::-1])
    