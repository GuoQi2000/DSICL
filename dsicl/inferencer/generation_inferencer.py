import numpy as np
import torch
from typing import List, Tuple, Dict
from tqdm import trange
from dsicl.data_reader import DataReader
from dsicl.prompter import Prompter
from dsicl.inferencer.base_inferencer import BaseInferencer
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, \
    STOPPING_CRITERIA_INPUTS_DOCSTRING, add_start_docstrings

class StopAtSpecificTokenCriteria(StoppingCriteria):
    """
    当生成出第一个指定token时，立即停止生成
    ---------------
    ver: 2023-08-02
    by: changhongyu
    """
    def __init__(self, token_id_list: List[int] = None):
        """
        :param token_id_list: 停止生成的指定token的id的列表
        """
        self.token_id_list = token_id_list
        
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # return np.argmax(scores[-1].detach().cpu().numpy()) in self.token_id_list
        # 储存scores会额外占用资源，所以直接用input_ids进行判断
        return input_ids[0][-1].detach().cpu().numpy() in self.token_id_list

class GenerationInferencer(BaseInferencer):

    def __init__(self, model, tokenizer, prompter: Prompter, stop_tokens = ['\n']) -> None:
        super(BaseInferencer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompter = prompter

        self.stopping_criteria = StoppingCriteriaList()
        self.stop_token_id = [self.tokenizer.encode(_)[-1] for _ in stop_tokens]
        self.stopping_criteria.append(StopAtSpecificTokenCriteria(token_id_list=self.stop_token_id))


    def infer(self, demos: DataReader, sample: Dict):
        with torch.no_grad():
            context = self.prompter.generate_context(demos, sample)

            inputs = self.tokenizer(context, return_tensors="pt").to(self.model.device)
            l = inputs.input_ids.size()[1]
            outputs = self.model.generate(**inputs, max_new_tokens=20, stopping_criteria=self.stopping_criteria)[0]
            return self.tokenizer.decode(outputs[l:-1])
    
    def batch_infer(self, demos_l: List[DataReader] , data: DataReader) -> List[str]:
        return [self.infer(demos_l[_], data[_]) for _ in trange(len(data))]


        