from typing import List, Tuple, Dict
import torch
import numpy as np

class BaseSelector():
    def __init__(self) -> None:
        pass

    def select(self, data: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        return data