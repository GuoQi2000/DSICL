import json
from typing import List, Union, Optional, Dict, Tuple
import logging
import numpy as np
from .template import LABEL_MAP
from .utils import set_seed
# DATA_ROOT = "/home/gq/DS-ICL_benchmark/"
DATA_ROOT = "/home/gq/ICL_data/"

def read_train_test(task, seed):
    set_seed(seed)
    if task == 'rte':
        trainset = get_data_reader(file_name='rte_train.json',  label_map={'entailment': 'yes','not_entailment':'no'})
        testset1 = get_data_reader(file_name='scitail.json', label_map={0: 'yes',1:'no'})
        testset2 = get_data_reader(file_name='hans.json', label_map={0: 'yes',1:'no'})
        testset = DataReader(testset1.data+testset2.data)
    elif task == 'sst2':
        trainset = get_data_reader(file_name='sst2.json',  label_map={0: 'negative',1:'positive'})
        testset1 = get_data_reader(file_name='flipkart.json', label_map={0: 'negative',1:'positive'})
        testset2 = get_data_reader(file_name='yelp_polarity.json', label_map={0: 'negative',1:'positive'})
        testset3 = get_data_reader(file_name='imdb.json', label_map={0: 'negative',1:'positive'})
        testset = DataReader(testset1.data+testset2.data+testset3.data)
    elif task == 'mnli':
        trainset = get_data_reader(file_name='mnli.json',  label_map={0: 'yes',1:'maybe',2:'no'})
        testset1 = get_data_reader(file_name='mnli_mis.json', label_map={0: 'yes',1:'maybe',2:'no'})
        testset2 = get_data_reader(file_name='snli.json', label_map={0: 'yes',1:'maybe',2:'no'})
        testset = DataReader(testset1.data+testset2.data)
    elif task == 'qnli':
        trainset = get_data_reader(file_name='qnli.json',  label_map={0: 'yes',1:'no'})
        testset = get_data_reader(file_name='newsqa.json', label_map={0: 'no',1:'yes'})
    elif task == 'mrpc':
        trainset = get_data_reader(file_name='mrpc.json',  label_map={0: 'no',1:'yes'})
        testset1 = get_data_reader(file_name='qqp.json',  label_map={0: 'no',1:'yes'})
        testset2 = get_data_reader(file_name='twitter.json', label_map={0: 'no',1:'yes'})
        testset = DataReader(testset1.data+testset2.data)
    elif task == 'cola':
        trainset = get_data_reader(file_name='cola.json',  label_map={0: 'no',1:'yes'})
        testset = get_data_reader(file_name='cola_ood.json', label_map={0: 'no',1:'yes'})
    testset, _ = testset.split(num=min(len(testset), 3000), random=True)
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
        if not label_map is None:
            for d in self.data:
                d['label'] = label_map[d['label']]
    def __getitem__(self, index: int):
        
        return self.data[index]
    
    def to_list(self):
        return self.data

    def add(self, data: Dict):
        self.data.append(data)

    def get_label(self, index: int):
        return self.__getitem__(index)['label']
    
    def random_split(self, ratio: float = None, num: int = None) -> Tuple:
        random_indexs = np.random.choice(self.__len__(), self.__len__(), replace=False)
        if ratio:
            split_point = int(self.__len__()*ratio)
        elif num:
            split_point = num
        else:
            raise Exception("Sorry, ratio or num should be set.")
        
        return DataReader([self.data[random_indexs[_]] for _ in range(split_point)], self.data_info), \
                DataReader([self.data[random_indexs[_]] for _ in range(split_point, self.__len__())], self.data_info)

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

    def get_subset(self, nums):
        shuffled_index = np.arange(self.__len__())
        np.random.shuffle(shuffled_index)
        return [self.__getitem__(shuffled_index[i]) for i in range(0, nums)]

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

