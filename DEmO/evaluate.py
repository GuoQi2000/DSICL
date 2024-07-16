import torch
import json
import os
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.append('../../DSICL/')

from dsicl.utils import set_seed
from dsicl.data_reader import read_demo_benchmark
from dsicl.prompter import Prompter
from dsicl.template import DEMO_TEMPLATE, DEMO_HEAD
from dsicl.inferencer import DirectInferencer
from dsicl.evaluator import Evaluator
import argparse

def paras_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu',default='0', type=str, help='Specify the GPU to use')
    parser.add_argument('--saving_path',default='./output/', type=str, help='Directory to save the demonstrations')
    parser.add_argument('--model_path',default='', type=str, help='Model to use')
    parser.add_argument('--task',default='', type=str, help='The task to evaluate')
    parser.add_argument('--seed',default='', type=int, help='Random seed')
    parser.add_argument('--method',default='', type=str, help='Method to evaluate')
    args = parser.parse_args()
    return args

def load_sample_demos_pairs(saving_path):
    with open(f'{saving_path}/demos.json', 'r') as f:
        sample_demos_pairs = json.load(f)
    return sample_demos_pairs

def main(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, legacy=False, use_fast=False, padding_side="right")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, use_cache=True).half().to(torch.device('cuda'))

    if not os.path.exists(args.saving_path):
        os.mkdir(args.saving_path)

    sample_demos_pairs = load_sample_demos_pairs(args.saving_path)
    
    samples = [sample_demos_pair['sample'] for sample_demos_pair in sample_demos_pairs]

    demos_l = [sample_demos_pair['demos'] for sample_demos_pair in sample_demos_pairs]

    template, head = DEMO_TEMPLATE[args.task], DEMO_HEAD[args.task]

    prompter = Prompter(template=template, head=head, sep='\n')

    labels = read_demo_benchmark(task=args.task, seed=0)[0].data_info['label_space']

    direct_inferencer = DirectInferencer(model, tokenizer, prompter, labels)

    y_p = direct_inferencer.batch_infer(demos_l, samples)

    y_t = [sample['label'] for sample in samples]

    with open(f'{args.saving_path}/preds.json', 'w') as f:
        preds = [{'y_t':y_t[i], 'y_p':y_p[i]} for i in range(len(samples))]
        json.dump(preds, f, indent=4)

    evaluator = Evaluator()

    acc = evaluator.acc_evaluate(y_p, y_t)

    with open(f'{args.saving_path}/evaluation.json', 'w') as f:
        json.dump({f'Accuracy':acc}, f, indent=4)
    
    print(f'Accuracy of {args.method}: {acc:.4f}')

if __name__ == '__main__':
    args = paras_args()
    main(args)
