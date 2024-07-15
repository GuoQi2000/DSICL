import numpy as np
import torch
from typing import List, Tuple, Dict, Any
from dsicl.data_reader import DataReader
from dsicl.ranker.base_ranker import BaseRanker
from dsicl.model import speculative_decoder
from dsicl.prompter import Prompter
from tqdm import trange, tqdm
from dsicl.utils import set_seed

class DEmORanker(BaseRanker):

    def __init__(self, model, tokenizer, prompter: Prompter, labels, content_free_tokens= ['[N/A]','','[MASK]'], candidate_num = 4, iteration_num =100) -> None:
        super(BaseRanker, self).__init__()
        
        self.prompter = prompter
        self.decoder = speculative_decoder(model, tokenizer)
        self.candidate_num = candidate_num
        self.iteration_num = iteration_num
        self.labels = labels
        self.content_free_tokens = content_free_tokens
        self.candidate_demos = None

    def entropy(self, probs: torch.Tensor) -> torch.Tensor:
        return torch.sum(-probs*torch.log2(probs+1e-10))

    def stage_1(self, demos: List[Dict[str, Any]], orders: List[int]) -> Dict[str, Any]:
        content_free_sample = {k:v for k,v in demos[0].items()}

        content_free_entropy_l = []
        content_free_probs_l = []
        for order in tqdm(orders):
            content_free_probs = torch.zeros(len(self.labels)).to(self.decoder.device)
            for token in self.content_free_tokens:
                for key, value in demos[0].items():
                    if not key == 'label':
                        content_free_sample[key] = token
                prefix = self.prompter.generate_context([demos[_] for _ in order], content_free_sample)
                probs = self.decoder.decode(prefix, self.labels)
                content_free_probs += probs
            content_free_probs /= len(self.content_free_tokens)

            content_free_entropy_l.append(self.entropy(content_free_probs))
            content_free_probs_l.append(content_free_probs)
        indexs = torch.argsort(-torch.tensor(content_free_entropy_l))[:self.candidate_num]
        return [[demos[o] for o in orders[_]] for _ in indexs],   [content_free_probs_l[_] for _ in indexs]

    def rank(self, demos: DataReader, sample: Dict, shots: int) -> DataReader:
        if not self.candidate_demos:
            candidate_orders = [np.random.choice(len(demos),shots,replace=False) for _ in range(self.iteration_num)]
            self.candidate_demos, self.candidate_bias = self.stage_1(demos, candidate_orders)
        
        influence_l = []
        for i, d in enumerate(self.candidate_demos):
            prefix = self.prompter.generate_context(d,sample)
            probs = self.decoder.decode(prefix, self.labels)
            influence = probs.max() - self.candidate_bias[i][probs.argmax()]
            influence_l.append(influence)

        return self.candidate_demos[torch.tensor(influence_l).argmax()]
            

    def batch_rank(self, demos_l: List[DataReader] , data: DataReader, shots: int) -> List[DataReader]:
        return [self.rank(demos=demos_l[_], sample= data[_], shots=shots) for _ in trange(len(data))]