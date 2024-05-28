import numpy as np
import os
from src.model import get_encoder
from tqdm import trange
import torch
from typing import List, Tuple, Dict, Union

class BaseRetriever():
    def __init__(self) -> None:
        pass

    def retrieve(self, x: str, data: List[Tuple[str, str]], k: int) -> None:
        return data
    
# class random_retriever(retriever):
#     def __init__(self) -> None:
#         super(retriever, self).__init__()

#     def retrieve(self, x: str, data: List[Tuple[str, str]], k: int) -> List[Tuple[str, str]]:
#         idxs = np.random.choice(len(data), k, replace=False)
#         return [data[_] for _ in idxs]
    
# class embedding_retriever(retriever):
#     def __init__(self, paras: dict) -> None:
#         super(retriever, self).__init__()
#         self.paras = paras
#         if paras['encoder'] == 'roberta-large':
#             self.encoder = get_encoder('roberta-large', device=torch.device('cuda'))

#         self.embedding_file = f"embedding/{self.paras['task']}_{self.paras['encoder']}.pt"
#         if not os.path.exists(self.embedding_file):
#             print('build embeddings')
#             self.build_embeddings(paras['data'])
        
#         self.embeddings = torch.load(self.embedding_file)
            
#     def encode(self, x: str) -> torch.FloatTensor:
#         return self.encoder(x)

#     def build_embeddings(self, data: list) -> None:
#         embeddings = []
#         for i in trange(len(data)):
#             x = data[i][0]
#             v = self.encode(x)
#             embeddings.append(v)
#         embeddings = torch.stack(embeddings)
#         torch.save(embeddings, f"embedding/{self.paras['task']}_{self.paras['encoder']}.pt")

#     def retrieve(self, x: str, data: List[Tuple[str, str]], k: int) -> List[Tuple[str, str]]:
#         q = self.encode(x)
#         distances = torch.sum((q-self.embeddings)**2, dim=1)**0.5
#         idxs = distances.argsort()[:k]
#         return [data[_] for _ in idxs][::-1]
    
# def get_retriever(method, para_dict: dict = None) -> retriever:
#     if method == 'random':
#         return random_retriever()
#     elif method == 'kate':
#         return embedding_retriever(para_dict)
#     elif method == 'adaptive':
#         return embedding_retriever(para_dict)
#     elif method == 'gle':
#         return retriever()
#     elif method == 'mi':
#         return retriever()
#     else:
#         print(f'wrong method: {method}')
