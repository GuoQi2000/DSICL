import torch
import numpy as np
import json
from typing import List, Tuple, Dict
from tqdm import trange, tqdm
from collections import defaultdict

class Evaluator:
    def __init__(self) -> None:
        pass

    def acc_evaluate(self, y_pred: List[str], y_true: List[str]) -> float:
        acc = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_pred)
        # if confusion_matrix:
            
        return float(acc)
    
    def f1_evaluate(self, y_pred: List[str], y_true: List[str]) -> float:
        # 计算每个类别的 TP, FP, FN
        true_positives = defaultdict(int)
        false_positives = defaultdict(int)
        false_negatives = defaultdict(int)

        for yp, yt in zip(y_pred, y_true):
            if yp == yt:
                true_positives[yp] += 1
            else:
                false_positives[yp] += 1
                false_negatives[yt] += 1
        
        # 计算每个类别的 F1 分数
        f1_scores = []
        for label in set(y_pred).union(set(y_true)):
            tp = true_positives[label]
            fp = false_positives[label]
            fn = false_negatives[label]
            
            # 避免除以零
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            f1_scores.append(f1)
        
        # 计算宏平均 F1 分数
        if f1_scores:
            return sum(f1_scores) / len(f1_scores)
        else:
            return 0.0

    def std(self, values: List[float]) -> float:
        if not values:
            return 0.0  # 空列表的标准差定义为 0

        # 计算均值
        mean = sum(values) / len(values)
        
        # 计算方差
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        
        # 计算标准差
        return float((np.array(variance)**0.5))
    
    def mean(self, values: List[float]) -> float:
        if not values:
            return 0.0  # 空列表的均值定义为 0

        # 计算均值
        mean = sum(values) / len(values)
        
        return mean


            
        