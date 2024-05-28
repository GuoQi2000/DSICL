import numpy as np
import os
from src.model import get_encoder
from tqdm import trange, tqdm
import torch
from typing import List, Tuple, Dict, Union
from src.retriever.base_retriever import BaseRetriever
from src.data_reader import DataReader
from src.prompter import Prompter

class EmbeddingRetriever(BaseRetriever):

    def __init__(self, 
                 encoder: str,
                 prompter: Prompter,
                 data: DataReader = None,
                 embedding_path: str = None) -> None:
        
        super(BaseRetriever, self).__init__()

        self.encoder = get_encoder(encoder, device=torch.device('cuda'))
        self.prompter = prompter

        if not data is None:
            self.data = data
            self.embeddings = None

        if not embedding_path is None:
            self.embeddings = torch.load(embedding_path)
        
            
    def encode(self, x: str) -> torch.FloatTensor:
        return self.encoder(x)

    def save_embeddings(self, path):
        # torch.save(self.embeddings, f"embedding/{self.paras['task']}_{self.paras['encoder']}.pt")
        torch.save(self.embeddings, path)

    def build_embeddings(self) -> None:
        embeddings = []
        for i in trange(len(self.data)):
            x = self.prompter.generate_prompt(self.data[i])
            v = self.encode(x)
            embeddings.append(v)
        self.embeddings = torch.stack(embeddings)
#        torch.save(embeddings, f"embedding/{self.paras['task']}_{self.paras['encoder']}.pt")

    def retrieve(self, sample: Dict, k: int) -> DataReader:
        q = self.encode(self.prompter.generate_prompt(sample))
        distances = torch.sum((q-self.embeddings)**2, dim=1)**0.5
        idxs = distances.argsort()[:k]
        return DataReader([self.data[_] for _ in idxs][::-1])
    
    def batch_retrieve(self, data: DataReader, k: int) -> List[DataReader]:

        return [self.retrieve(d,k) for d in tqdm(data)]
        # q = self.encode(self.prompter.generate_prompt(sample))
        # distances = torch.sum((q-self.embeddings)**2, dim=1)**0.5
        # idxs = distances.argsort()[:k]
        # return DataReader([self.data[_] for _ in idxs][::-1])
    