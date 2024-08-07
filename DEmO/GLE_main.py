import torch
import json
import os
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.append('../../DSICL/')

from dsicl.utils import set_seed
from dsicl.data_reader import DataReader, read_demo_benchmark
from dsicl.prompter import Prompter
from dsicl.template import DEMO_TEMPLATE, DEMO_HEAD
from dsicl.selector import GLESelector
import argparse

def paras_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu',default='0', type=str, help='Specify the GPU to use')
    parser.add_argument('--saving_path',default='./output/', type=str, help='Directory to save the demonstrations')
    parser.add_argument('--model_path',default='', type=str, help='Model to use')
    parser.add_argument('--data_path',default='', type=str, help='Path of DEmO data')
    parser.add_argument('--task',default='', type=str, help='The task to evaluate')
    parser.add_argument('--seed',default='', type=int, help='Random seed')
    parser.add_argument('--shots',default=4, type=int, help='Number of demonstrations')
    parser.add_argument('--candidate_num',default=4, type=int, help='Number of candidate orders')
    parser.add_argument('--val_size',default=100, type=int, help='Size of validation set')
    args = parser.parse_args()
    return args

def main(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, legacy=False, use_fast=False, padding_side="right")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, use_cache=True).half().to(torch.device('cuda'))

    if not os.path.exists(args.saving_path):
        os.makedirs(args.saving_path)

    trainset, testset = read_demo_benchmark(data_path=args.data_path, task=args.task, seed=args.seed)

    template, head = DEMO_TEMPLATE[args.task], DEMO_HEAD[args.task]

    prompter = Prompter(template=template, head=head, sep='\n')

    set_seed(args.seed)
    
    original_demos, others = trainset.get_subset(len(trainset.data_info['label_space'])*args.shots, balance=True, return_left=True)

    val_data = others.get_subset(args.val_size)

    selector = GLESelector(model, tokenizer, prompter, trainset.data_info['label_space'], candidate_num=args.candidate_num)

    demos_l = [selector.select(original_demos, len(original_demos), val_data).to_list()] * len(testset)

    with open(f'{args.saving_path}/demos.json', 'w') as f:
        sample_demos_pairs = [{'sample':testset[i], 'demos':demos_l[i]} for i in range(len(testset))]
        json.dump(sample_demos_pairs, f, indent=4)

    # Save arguments

    with open(f'{args.saving_path}/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)


if __name__ == '__main__':
    args = paras_args()
    main(args)
