import json
from typing import List, Union, Optional, Dict, Tuple
import logging
import numpy as np
from .template import LABEL_MAP
from .utils import set_seed

DATA_ROOT = "../DEmO/DEmO_data/"

def read_demo_benchmark(task, seed):
    if task == 'sst2':
        trainset = get_data_reader(file_name='sst2_train.json', label_map={})
        testset = get_data_reader(file_name='sst2_dev.json', label_map={})
    elif task == 'cr':
        trainset = get_data_reader(file_name='cr_train.json', label_map={})
        testset = get_data_reader(file_name='cr_test.json', label_map={})
    elif task == 'mr':
        trainset = get_data_reader(file_name='mr_train.json', label_map={})
        testset = get_data_reader(file_name='mr_test.json', label_map={})
    elif task == 'subj':
        trainset = get_data_reader(file_name='subj_train.json', label_map={})
        testset = get_data_reader(file_name='subj_test.json', label_map={})
    elif task == 'rte':
        trainset = get_data_reader(file_name='rte_train.json', label_map={})
        testset = get_data_reader(file_name='rte_val.json', label_map={})
    elif task == 'snli':
        trainset = get_data_reader(file_name='snli_train.json', label_map={})
        testset = get_data_reader(file_name='snli_test.json', label_map={})
    elif task == 'agnews':
        trainset = get_data_reader(file_name='agnews_train.json', label_map={})
        testset = get_data_reader(file_name='agnews_test.json', label_map={})
    elif task == 'trec':
        trainset = get_data_reader(file_name='trec_train.json', label_map={})
        testset = get_data_reader(file_name='trec_test.json', label_map={})
    elif task == 'dbpedia':
        trainset = get_data_reader(file_name='dbpedia_train.json', label_map={})
        testset = get_data_reader(file_name='dbpedia_test.json', label_map={})
    else:
        raise Exception("Sorry, the task is not supported.")
    set_seed(seed)
    testset = testset.get_subset(256)
    return trainset, testset

class DataReader: ## sentence1  sentence2  gold_label

    """ICL Data Reader Class
        Generate an DataReader instance provided with data file in a json format.
        
    Attributes:
        data (:List[Dict]): 数据列表: 每个数据由一个字典构成
        data_info (: Dict): 数据集基本信息: 包括数据集名称、数据属性名称、标签空间.
    """

    def __init__(self, 
                data: List[Dict],
                data_info:Dict = None,
                label_map = None
                ):
        self.data = data
        self.data_info = data_info
        if not label_map is None and not label_map == {}:
            for d in self.data:
                d['label'] = label_map[d['label']]
                
    def __getitem__(self, index: int):
        
        return self.data[index]
    
    def to_list(self):
        return self.data

    def split(self, ratio: float = None, num: int = None, random = False) -> Tuple:
        idxs = [i for i in range(len(self.data))]
        if random:
            np.random.shuffle(idxs)
        if ratio:
            split_point = int(self.__len__()*ratio)
        elif num:
            split_point = num
        else:
            raise Exception("Sorry, ratio or num should be set.")
        
        return DataReader([self.data[idxs[_]] for _ in range(split_point)], self.data_info), \
                DataReader([self.data[idxs[_]] for _ in range(split_point, self.__len__())], self.data_info)

    def get_subset(self, nums, balance=False, return_left = False):
        subset = []
        if balance:
            labels = {}
            
            # Count the number of samples for each label
            for i, sample in enumerate(self.data):
                label = sample['label']
                if not label in labels:
                    labels[label] = []
                labels[label].append(i)
            
            # Extract balanced samples

            for key in labels.keys():
                indexs = np.random.choice(labels[key], int(nums/len(labels.keys())),replace=False)
                subset.extend([self.data[_] for _ in indexs])
                np.random.shuffle(subset)
        else:
            indexs = np.random.choice(len(self.data), nums, replace=False)
            subset.extend([self.data[indexs[i]] for i in range(nums)])
        if return_left:
            left = [self.data[i] for i in range(len(self.data)) if not i in indexs]
            return DataReader(subset, self.data_info), DataReader(left, self.data_info)
        else:
            return DataReader(subset, self.data_info)
    
    def __len__(self):
        return len(self.data)
    
    def info_display(self):
        print("{:=^50s}".format(self.data_info['data_name']))
        print(f"data_size : {self.__len__()}")
        print(f"example   : {self.__getitem__(0)}")
        print(f"columns : {self.data_info['columns']}")
        print(f"label space : {self.data_info['label_space']}")

def get_data_reader(file_name, label_map=None, data_root = None) -> DataReader:
    data = None
    if label_map is None:
        label_map = LABEL_MAP[file_name.split('/')[0]]
    if not data_root is None:
        file_path = data_root + file_name
    else:
        file_path = DATA_ROOT + file_name
    with open(file_path,'r') as f:
        data_dic = json.load(f)
        data = DataReader(data_dic['data'], data_dic['data_info'], label_map)
        f.close()
    return data

