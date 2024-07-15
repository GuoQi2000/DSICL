import numpy as np
import torch
from typing import List, Tuple, Dict

class BaseRanker():
    def __init__(self) -> None:
        pass

    def rank(self, x: str, demo: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        return demo