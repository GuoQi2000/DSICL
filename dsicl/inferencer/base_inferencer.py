import numpy as np
import torch
from typing import List, Tuple, Dict

class BaseInferencer():
    def __init__(self) -> None:
        pass

    def infer(self, demos: List[Tuple[str, str]], sample: Dict) -> List[Tuple[str, str]]:
        return None