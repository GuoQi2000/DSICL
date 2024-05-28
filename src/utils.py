import torch
import json
import random
import numpy as np
from typing import List, Tuple, Dict

DATA_ROOT = "/home/gq/benchmark/"

def set_seed(seed):
    # Set the random seed for NumPy
    np.random.seed(seed)
    
    # Set the random seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        
    # Set cudnn.benchmark to False to ensure reproducibility
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_data_file(file: str) -> List[Dict]:
    content = None
    with open(DATA_ROOT+file,'r', encoding='utf-8') as f:
        content = json.load(f)
        f.close()
    return content['data'], content['data_info']

def read_samples(file):
    samples = [] 
    with open(file,'r') as f:
        for line in f.readlines():
            dic = json.loads(line)
            samples.append(dic['x'])
        f.close()
    return samples

def read_sample_labels(file):
    samples = [] 
    with open(file,'r') as f:
        for line in f.readlines():
            dic = json.loads(line)
            samples.append(dic['y'])
        f.close()
    return samples

def read_demos(file):
    demos = [] 
    with open(file,'r') as f:
        for line in f.readlines():
            dic = json.loads(line)
            demos.append(dic['demo'])
        f.close()
    return demos

def read_prompt(file):
    prompts = [] 
    with open(file,'r') as f:
        for line in f.readlines():
            dic = json.loads(line)
            prompts.append(dic['text'])
        f.close()
    return prompts

def read_output(file):
    yp = [] 
    with open(file,'r') as f:
        for line in f.readlines():
            dic = json.loads(line)
            yp.append(dic['output'])
        f.close()
    return yp

def write_demos(file, demos_l):
    with open(f'{file}','a+') as f:
        for i in range(len(demos_l)):
            dic_str = json.dumps({'id':i, 'demo':demos_l[i]})
            f.write(dic_str+'\n')
        f.close()

def write_prompt(file, prompts):
    with open(f'{file}','a+') as f:
        for i in range(len(prompts)):
            dic_str = json.dumps({'id':i, 'text':prompts[i]})
            f.write(dic_str+'\n')
        f.close()

def write_output(file, results):
    with open(f'{file}','a+') as f:
        for i in range(len(results)):
            dic_str = json.dumps({'id':i, 'output':results[i]})
            f.write(dic_str+'\n')
        f.close()

def save_demos(demos_l, testset, path):
    dic = {
        'demos':[_.to_list() for _ in demos_l],
        'data':[_ for _ in testset]
        }
    with open(path, 'a+') as f:
        f.write(json.dumps(dic,indent=4))
        f.close()
    
def read_demos(path):
    with open(path, 'r') as f:
        dic = json.loads(f.read())
        demos = dic['demos']
        data = dic['data']
        f.close()
    return demos, data

def save_preds(y_p, y_t, path):
    dic = {
        'y_p':y_p,
        'y_t':y_t
        }
    with open(path, 'a+') as f:
        f.write(json.dumps(dic,indent=4))
        f.close()

def read_preds(path):
    with open(path, 'r') as f:
        dic = json.loads(f.read())
        y_p = dic['y_p']
        y_t = dic['y_t']
        f.close()
    return y_p, y_t
    
def save_res(task, acc_mean, acc_std, f1_mean, f1_std, path):
    acc_mean = round(acc_mean, 4)
    acc_std = round(acc_std, 4)
    f1_mean = round(f1_mean, 4)
    f1_std = round(f1_std, 4)
    dic = {
        'task':task,
        'acc_mean':acc_mean,
        'acc_std':acc_std,
        'f1_mean':f1_mean,
        'f1_std':f1_std
        }
    with open(path, 'a+') as f:
        f.write(json.dumps(dic,indent=4)+'\n')
        f.close()