import time
import torch
import logging
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import RobertaTokenizer, RobertaModel
from accelerate import init_empty_weights,infer_auto_device_map,load_checkpoint_in_model,dispatch_model
from transformers import LlamaConfig,LlamaForCausalLM,LlamaTokenizer


MODEL_ROOT = '/home/gq/model/'

def get_tokenizer(name):
    if name == 'bert-base-uncased':
        return BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        print(f'invalid tokenizer {name}')

def get_LLM(model, device,half=False):
    return LLM(model,device=device,half=half)

class encoder(nn.Module):
    def __init__(self, model, device):
        super(encoder, self).__init__()
        self.device = device
        self.tokenizer = RobertaTokenizer.from_pretrained(MODEL_ROOT+model)
        self.encoder = RobertaModel.from_pretrained(MODEL_ROOT+model).to(device)

    def forward(self, x):
        with torch.no_grad(): 
            inputs = self.tokenizer(x, return_tensors='pt' ,max_length=512, truncation=True).to(self.device)
            output = self.encoder(**inputs)
            return output['pooler_output'][0]

def get_encoder(model, device):
    return encoder(model, device)

class LLM(nn.Module):
    def __init__(self,model_name, device, half=False) -> None:
        super(LLM, self).__init__()
        self.device = device
        model_path = ""
        if model_name == 'gpt-j':
            model_path = "gpt-j"
        if model_name == 'llama_1.3b':
            model_path = "llama_1.3b"
        if model_name == 'llama_2.7b':
            model_path = "llama_2.7b"
        if model_name == 'llama_13b':
            model_path = "llama_13b"
        if model_name == 'phi_2':
            model_path = "phi_2"
        if model_name == 'bloom':
            model_path = "bloom_1.7b"
        if model_name == 'opt_1.3b':
            model_path = "opt_1.3b"
        if model_name == 'gpt_j':
            model_path = "gpt-j"
        if model_name == 'gpt2_large':
            model_path = "models--gpt2-large/snapshots/97935fc1a406f447320c3db70fe9e9875dca2595/"
        if model_name == 'gpt2_xl':
            model_path = "gpt2_xl"
        if model_name == 'gpt2_medium':
            model_path = "models--gpt2-medium/snapshots/f65d4965d1221eff2bcf34f53a2ba12120e18f24/"
        # if model_name == "opt":
        #     model_path = "models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0/"
        if model_name == 'llama_7b':
            model_path = "llama_7b"
        if model_name == 'llama3_8b':
            model_path = 'llama3_8b'
        if model_name == 'gpt_neo_2.7b':
            model_path = "models--EleutherAI--gpt-neo-2.7B/snapshots/e24fa291132763e59f4a5422741b424fb5d59056/"
        if model_name == 'gpt_neo_1.3b':
            model_path = "models--EleutherAI--gpt-neo-1.3B/snapshots/8282180b53cba30a1575e49de1530019e5931739/"

        if model_name != 'llama_7b' and model_name != 'llama_13b' and model_name != 'gemma_2b': #and model_name != 'llama3_8b':
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT+model_path, padding_side='left')
            if half:
                self.model = AutoModelForCausalLM.from_pretrained(MODEL_ROOT+model_path).half().to(device)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(MODEL_ROOT+model_path).to(device)
        else:
            memory = '9GiB'
            cuda_list = '0,1'.split(',')
            no_split_module_classes = LlamaForCausalLM._no_split_modules
            max_memory = {int(cuda):memory for cuda in cuda_list}
            config = LlamaConfig.from_pretrained(MODEL_ROOT+model_path)
            with init_empty_weights():
                self.model = LlamaForCausalLM._from_config(config, torch_dtype=torch.float16) #加载到meta设备中，不需要耗时，不需要消耗内存和显存

            device_map = infer_auto_device_map(self.model, max_memory=max_memory,no_split_module_classes=no_split_module_classes) #自动划分每个层的设备
            load_checkpoint_in_model(self.model,MODEL_ROOT+model_path,device_map=device_map) #加载权重
            self.model = dispatch_model(self.model,device_map=device_map) #并分配到具体的设备上

            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT+model_path)
            torch.set_grad_enabled(False)
        self.model.eval()


class speculative_decoder:
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.device = self.model.device
        
    def decode(self, prefix:str, answer_list:list):
        truncation_l = 8000
        with torch.no_grad():

            log_soft = nn.LogSoftmax(dim=0)
            soft = nn.Softmax(dim=0)

            prompts = [prefix+answer for answer in answer_list] # 拼接所有候选label
            l = len(self.tokenizer(prefix,max_length=truncation_l,truncation=True, return_tensors="pt").input_ids[0]) # prefix长度

            inputs = self.tokenizer(prompts,max_length=truncation_l, truncation=True, padding= True, return_tensors="pt").to(self.model.device)
            seq_logits = self.model(**inputs).logits[:, l-1:-1,:]
            tokens = inputs.input_ids[:,l:] # 取label对应的logits
            # print(tokens)
            # print(inputs.input_ids[:,l-1:])
            log_probs = torch.zeros(len(prompts)).to(self.model.device) # 取概率的对数

            for i in range(len(prompts)):
                for j in range(seq_logits.size()[1]):
                    token_id = tokens[i][j]
                    if token_id == self.tokenizer.eos_token_id: # 遇到padding token结束
                        break
                    vocabulary = seq_logits[i][j]
                    log_probs[i] += log_soft(vocabulary)[token_id]

            probs = soft(log_probs)
            # print(probs)
            return probs
        
    