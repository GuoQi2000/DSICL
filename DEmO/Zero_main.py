import torch
import json
import os
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.append('../../DSICL/')

from dsicl.utils import set_seed
from dsicl.data_reader import get_data_reader, DataReader, read_demo_benchmark
import argparse

def paras_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu',default='0', type=str, help='Specify the GPU to use')
    parser.add_argument('--saving_path',default='./output/', type=str, help='Directory to save the demonstrations')
    parser.add_argument('--model_path',default='', type=str, help='Model to use')
    parser.add_argument('--task',default='', type=str, help='The task to evaluate')
    parser.add_argument('--seed',default='', type=int, help='Random seed')
    args = parser.parse_args()
    return args

def main(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.saving_path):
        os.makedirs(args.saving_path)

    trainset, testset = read_demo_benchmark(task=args.task, seed=args.seed)

    set_seed(args.seed)
    
    with open(f'{args.saving_path}/demos.json', 'w') as f:
        sample_demos_pairs = [{'sample':testset[i], 'demos':[]} for i in range(len(testset))]
        json.dump(sample_demos_pairs, f, indent=4)

    # Save arguments

    with open(f'{args.saving_path}/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)


if __name__ == '__main__':
    args = paras_args()
    main(args)
