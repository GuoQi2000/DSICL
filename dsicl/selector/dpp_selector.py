from typing import List, Tuple, Dict
import torch
from dppy.finite_dpps import FiniteDPP
from torch import Tensor
import numpy as np
from dsicl.data_reader import DataReader
from dsicl.selector.base_selector import BaseSelector
from dsicl.model import speculative_decoder
from dsicl.prompter import Prompter
from dsicl.model import get_encoder
from tqdm import trange

class DPPSelector(BaseSelector):
    def __init__(self, encoder, model, tokenizer, prompter: Prompter, labels, sem_size=100) -> None:

        super(BaseSelector, self).__init__()
        self.encoder = get_encoder(encoder, device=torch.device('cuda'))
        self.prompter = prompter
        self.decoder = speculative_decoder(model, tokenizer)
        self.labels = labels
        self.sem_size = sem_size

    def encode(self, x: str) -> torch.FloatTensor:
        s = self.encoder(x)
     #   torch.nn.functional.normalize(input, p=2.0, dim=1, eps=1e-12, out=None)
        return torch.nn.functional.normalize(s, p=2,eps=1e-12, dim=0)
    
    def get_influence(self, sample, val_data):

        inf = 0
        infs = []

        for j in range(len(val_data)):
            label_idx = self.labels.index(val_data[j]['label'])

            prefix = self.prompter.generate_context([sample], val_data[j])
            probs1 = self.decoder.decode(prefix, self.labels)
            
            prefix = self.prompter.generate_context([], val_data[j])
            probs2 = self.decoder.decode(prefix, self.labels)

            infs.append(probs1[label_idx] - probs2[label_idx])
            inf += probs1[label_idx] - probs2[label_idx]
        infs = torch.tensor(infs)

        return inf/len(val_data), infs    
    
    def select(self, data: DataReader, shots: int, val_data: DataReader) -> DataReader:
        # stage 1
        val_data, trainset = data.split(shots=100)
        S = [self.encode(self.prompter.generate_prompt(trainset[_])) for _ in trange(len(trainset))]
        S = torch.stack(S, dim=0)
        L_S = ((S)@(S.T)).cpu().numpy()
        DPP1 = FiniteDPP('likelihood',
                **{'L': L_S + np.eye(L_S.shape[0])})
        DPP1.flush_samples()
        sem_indexs = DPP1.sample_exact_k_dpp(size=shots)
        sem_demos = [trainset[_] for _ in sem_indexs]

        return DataReader(sem_demos)
        # stage 2
        Q = []
        E = []
        for i in range(len(sem_demos)):
            q, e = self.get_influence(sem_demos[i], val_data)
            Q.append(q)
            E.append(e)
        E = torch.stack(E,dim=0)
        Q = torch.tensor(Q)

        L = ((Q)*((E)@(E.T))*(Q.T)).cpu().numpy()
        DPP2 = FiniteDPP('likelihood',
                **{'L': L + np.eye(L.shape[0])})
        DPP2.flush_samples()
        dem_indexs = DPP2.sample_exact_k_dpp(size=num)
        return [sem_demos[_] for _ in dem_indexs]





        
        
        