import torch
import torch.nn as nn
import torch.nn.functional as F
from ProEmb.new_tg_ProGAT import ProGAT
from ProEmb.ProLSTM import BiLSTM
from ProEmb.new_tg_ProBert import model_bert, model_bert_raw
import numpy as np

class ProNetwork(nn.Module):

    def __init__(self, pro_input_size, pro_out_size, gat_size, seq_model_type, radius=2, T=1, p_dropout=0.3):
        super(ProNetwork, self).__init__()
        if seq_model_type == "bilstm":
            self.SeqEmb = BiLSTM(rnn_layers=1)
        elif seq_model_type == "bilstm-gat":
            self.SeqEmb = BiLSTM(rnn_layers=1)
        elif seq_model_type == "bert-nofinetune" or seq_model_type == "bert-finetune":
            self.SeqEmb = model_bert
        elif seq_model_type == "only_seq":
            self.SeqEmb = model_bert
        elif seq_model_type == "bert-nopretrain":
            self.SeqEmb = model_bert_raw

        self.seq_model_type = seq_model_type
        self.GatEmb = ProGAT(embedding_size=gat_size, radius=radius, T=T, p_dropout=p_dropout)


    def forward(self, tokenized_sent, amino_list, amino_degree_list, amino_mask):
        if self.seq_model_type == "bilstm":
            seq_bert_feature = self.SeqEmb(tokenized_sent)
        elif self.seq_model_type == "bilstm-gat":
            seq_bert_feature = self.SeqEmb(tokenized_sent)
            seq_gat_feature = self.GatEmb(amino_list, amino_degree_list, amino_mask)
        elif self.seq_model_type == "bert-nofinetune":
            self.SeqEmb.eval()
            with torch.no_grad():
                seq_bert_feature = self.SeqEmb(tokenized_sent)[1]
            seq_gat_feature = self.GatEmb(amino_list, amino_degree_list, amino_mask)
        elif self.seq_model_type == "bert-finetune":
            seq_bert_feature_all, seq_bert_feature = self.SeqEmb(tokenized_sent)
            device = amino_mask.device
            max_len = amino_list.shape[1]
            amino_mask_np = amino_mask.cpu().numpy()
            seq_bert_feature_all_np = seq_bert_feature_all.detach().cpu().numpy()
            amino_pretrain_emb = []
            for i, j in zip(amino_mask_np, seq_bert_feature_all_np):
                amino_mask_one = list(i)
                seq_bert_feature_one = list(j)
                if 0 in amino_mask_one:
                    mask_index = amino_mask_one.index(0)
                    seq_bert_feature_one = np.pad(seq_bert_feature_one[1: 1+mask_index], ((0, max_len-mask_index), (0, 0)), 'constant', constant_values=(0, 0))
                else:
                    seq_bert_feature_one = seq_bert_feature_one[1: 1+max_len]
                amino_pretrain_emb.append(seq_bert_feature_one)
            amino_pretrain_emb = torch.tensor(amino_pretrain_emb).to(device)
            seq_gat_feature = self.GatEmb(amino_pretrain_emb, amino_list, amino_degree_list, amino_mask)
        elif self.seq_model_type == "only_seq":
            seq_bert_feature = self.SeqEmb(tokenized_sent)[1]
        elif self.seq_model_type == "only_gat":
            seq_gat_feature = self.GatEmb(amino_list, amino_degree_list, amino_mask)
        elif self.seq_model_type == "bert-nopretrain":
            self.SeqEmb.eval()
            seq_bert_feature = self.SeqEmb(tokenized_sent)[1]
            seq_gat_feature = self.GatEmb(amino_list, amino_degree_list, amino_mask)

        if self.seq_model_type == "only_seq": 
            seq_feature = seq_bert_feature
        elif self.seq_model_type == "only_gat":
            seq_feature = seq_gat_feature
        elif self.seq_model_type == "bilstm":
            seq_feature = seq_bert_feature
        else:
            seq_feature = torch.cat([seq_bert_feature, seq_gat_feature], dim=1)

        return seq_feature

