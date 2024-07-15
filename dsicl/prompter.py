import json
from typing import List, Union, Optional, Dict, Tuple
import logging
import numpy as np
#from template import PROMPTING_HEAD, TEMPLATE, SENTENCE_HEAD
import functools

class Prompter:

    """In-context Learning Prompt Template Class
        This class represents a template that guides the generation of prompts in the retrieval or inference process.
        
    Attributes:
        template (:obj:`Dict` or :obj:`str`): A custom template dictionary or string. If a dictionary, the keys of the dictionary represent the values of the output_column, and the values represent the corresponding generated statement. If a string, it represents a string template. 
        column_token_map (:obj:`Dict`): A dictionary mapping column names to specific tokens. The tokens will be replaced by data in the corresponding column (one piece each time) during the retrieval or inference process.
        selected_column_name (:obj:`str`, optional): Used only with string-type templates. A specific column that needs its value to be mapped.
        selected_column_map (:obj:`Dict`, optional): Used only with string-type templates. Maps the value of the column :obj:`selected_column_name`.
        ice_token(:obj:`str`, optional): A string that represents the specific token mapping from in-context examples. None if you want to use this template only to generate in-context examples, otherwise it can be used to generate the final prompt that is fed into the PLM. The ice_token will be invisible when generating in-context examples.
    """

    def __init__(self, template: str, head: str, sep: str = '\n', label_map = None) -> None:
        self.template = template
        self.prompting_head = head
        self.sep = sep
        self.label_map = label_map

    def generate_label_prompt(self, demo: Dict):
        if self.label_map is None:
            prompt = prompt = self.template.replace('[label]',demo['label'])
        else:
            prompt = prompt = self.template.replace('[label]',self.label_map[demo['label']])
        for key, value in demo.items():
            prompt = prompt.replace(f'[{key}]', str(value))
        return prompt

    def generate_prompt(self, sample: Dict):
        prompt = self.template.replace('[label]','')
        for key, value in sample.items():
            prompt = prompt.replace(f'[{key}]', str(value))
        return prompt

    def generate_context(self, demonstrations: List[Dict], sample: Dict):
        context = self.prompting_head+self.sep
        for demo in demonstrations:
            context += self.generate_label_prompt(demo)
            context += self.sep
        context += self.generate_prompt(sample)
        return context

