from transformers import AutoModel, AutoConfig, BertTokenizer, AutoModelWithLMHead
import torch
import os
import copy

gpu_start = 0
config_name = "./PPI/drug/seq/Pretraining-Yourself-Bert-From-Scratch/lm_smallBert/bert_config.json"
config = AutoConfig.for_model('bert').from_json_file(config_name)
device = torch.device("cuda:{}".format(gpu_start) if torch.cuda.is_available() else "cpu")

model_mlm = AutoModelWithLMHead.from_config(config)
model_mlm = model_mlm.to(device)
model_mlm = torch.nn.DataParallel(model_mlm, device_ids=[gpu_start, gpu_start+1, gpu_start+2, gpu_start+3])
model_mlm.load_state_dict(torch.load(os.path.join('./PPI/drug/seq/Pretraining-Yourself-Bert-From-Scratch/lm_smallBert/outputs_mlm/', "pytorch_model.pth")))
model_mlm = model_mlm.module  # 转为单gpu
model_mlm_dict = model_mlm.state_dict()

model_bert = AutoModel.from_config(config)
model_bert_raw = copy.deepcopy(model_bert)
model_bert_dict = model_bert.state_dict()

model_mlm_dict =  {k: v for k, v in model_mlm_dict.items() if k in model_bert_dict}
model_bert_dict.update(model_mlm_dict)
model_bert.load_state_dict(model_bert_dict)

# for param_name, param_value in model_mlm.named_parameters():
#     print(param_name, ":", param_value.size())
# print("-------------------------------------------------------------")
# for param_name, param_value in model_bert.named_parameters():
#     print(param_name, ":", param_value.size())