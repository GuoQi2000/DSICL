import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.data_reader import get_data_reader, DataReader
from src.prompter import Prompter
from src.selector import RandomSelector
from src.inferencer import DirectInferencer
from src.evaluator import Evaluator

# 换成本地的model地址或默认的hugging face模型名称
model_path = "../model/llama_2.7b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).half().to(torch.device('cuda'))

# 读取数据集，用get_data_reader方法从一个json文件中读取，json文件格式要求如下：
# { \
#    'data_info':{\
#     'data_name': 'rte_train',\
#     'label_space': ['entailment', 'not_entailment'],\
#     'columns': ['premise', 'hypothesis', 'label']\
#    },\
#    'data':[\
#     sample1,\
#     sample2,\
#     ...\
#    ]\
# }\
# \
# 读取数据时传入label_map参数，将标签从默认标签转换为想要的标签
# 其中每个sample是一个字典，由若干关键词（必须包括label）构成，例如：
# {'premise': 'No Weapons of Mass Destruction Found in Iraq Yet.',
#  'hypothesis': 'Weapons of Mass Destruction Found in Iraq.',
#  'label': 'no'}
trainset = get_data_reader(file_name='rte_train.json',  label_map={'entailment': 'yes','not_entailment':'no'})
testset = get_data_reader(file_name='scitail.json', label_map={0: 'yes',1:'no'})

# 初始化一个prompter用于生成上下文，需要给定一个模板（必须），一个提示头（可选），模板中将需要替换的sample关键词用[]标出
template = "Premise:[premise]\nHypothesis:[hypothesis]\nAnswer:[label]"

prompter = Prompter(template=template, head="This is a head", sep='\n')

# prompter的generate_context方法将demos和sample处理成prompt
# print(prompter.generate_context(trainset[:3], testset[0]))

# 挑选示例的方法主要由下面三个类实现，其中大部分算法由前两种实现。
# 1. selector: 从一个数据集中挑选出一组示例
# 2. retriever: 为一个sample，从数据集中检索出一组示例
# 3. ranker: 根据一个sample，为一组示例进行排序

random_selector = RandomSelector()
demos_l = [random_selector.select(trainset,num=8) for _ in range(len(testset))]

# 初始化一个inferencer进行推理，目前支持direct inferencer（利用推测解码直接获取label上的概率）和generation inferencer（常规的自回归生成）

labels = ['yes','no','maybe']
direct_inferencer = DirectInferencer(model, tokenizer, prompter, labels)

# 进行推理

y_p = direct_inferencer.batch_infer(demos_l, testset[:100])

# 可以调用Evaluator类对结果进行评估（暂时只支持accuracy和f1-score，可以自己拓展）
y_t = [testset[_]['label'] for _ in range(len(testset[:100]))]
evaluator = Evaluator()
print(evaluator.acc_evaluate(y_p, y_t))