from transformers import AutoModel, AutoConfig, BertTokenizer, AutoModelWithLMHead
import torch
import os
import copy

config_name = "./pretrained_model/bert_config.json"
model_name = "./pretrained_model/pytorch_model.pth"
pretrained_state_dict = torch.load(model_name, map_location='cpu')
config = AutoConfig.for_model('bert').from_json_file(config_name)

model_bert = AutoModel.from_config(config)
model_bert_raw = copy.deepcopy(model_bert)
model_bert_dict = model_bert.state_dict()

state_dict_common = {}
for k, v in pretrained_state_dict.items():
    k = k.replace('module.', '')
    if k in model_bert_dict:
        state_dict_common.update({k: v})

model_bert_dict.update(state_dict_common)
model_bert.load_state_dict(model_bert_dict)