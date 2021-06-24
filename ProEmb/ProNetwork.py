import torch
import torch.nn as nn
import torch.nn.functional as F
from ProEmb.ProGAT import ProGAT
from ProEmb.ProLSTM import BiLSTM
from ProEmb.ProBert import model_bert, model_bert_raw
# from ProEmb.ProBert_tape import model_bert, model_bert_raw

class ProNetwork(nn.Module):

    def __init__(self, pro_size, seq_size, gat_size, seq_model_type, radius, T, p_dropout):
        super(ProNetwork, self).__init__()

        if seq_model_type == "bilstm":
            self.SeqEmb = BiLSTM(rnn_layers=1)
            self.GatEmb = ProGAT(embedding_size=gat_size, radius=radius, T=T, p_dropout=p_dropout)
        elif seq_model_type == "bilstm-gat":
            self.SeqEmb = BiLSTM(rnn_layers=1)
            self.GatEmb = ProGAT(embedding_size=gat_size, radius=radius, T=T, p_dropout=p_dropout)
        elif seq_model_type == "bert-nofinetune" or seq_model_type == "bert-finetune":
            self.SeqEmb = model_bert
            self.GatEmb = ProGAT(embedding_size=gat_size, radius=radius, T=T, p_dropout=p_dropout)
        elif seq_model_type == "only_seq":
            self.SeqEmb = model_bert
        elif seq_model_type == "only_gat":
            self.GatEmb = ProGAT(embedding_size=gat_size, radius=radius, T=T, p_dropout=p_dropout)
        elif seq_model_type == "bert-nopretrain":
            self.SeqEmb = model_bert_raw
            self.GatEmb = ProGAT(embedding_size=gat_size, radius=radius, T=T, p_dropout=p_dropout)
        elif seq_model_type == "linear-aug-norm":
            self.SeqEmb = model_bert
            self.seq_layernorm = nn.LayerNorm(seq_size)
            self.gat_layernorm = nn.LayerNorm(gat_size)
            self.predict_n = nn.Sequential(nn.Dropout(p_dropout),
                                           nn.Linear(seq_size+gat_size, pro_size), 
                                           nn.ReLU())
            self.GatEmb = ProGAT(embedding_size=gat_size, radius=radius, T=T, p_dropout=p_dropout)

        self.seq_model_type = seq_model_type


    def forward(self, amino_list=[], amino_degree_list=[], amino_mask=[], 
                tokenized_sent=[], attention_mask=[], token_type_ids=[], is_auxiliary=False, is_auxiliary_dssp=False):
        if is_auxiliary:
            seq_bert_feature = self.SeqEmb(tokenized_sent)[1]
            return seq_bert_feature
        
        if is_auxiliary_dssp:
            seq_bert_feature = self.SeqEmb(tokenized_sent)[0]
            return seq_bert_feature

        if self.seq_model_type == "bilstm":
            seq_bert_feature = self.SeqEmb(tokenized_sent)
        elif self.seq_model_type == "bilstm-gat":
            seq_bert_feature = self.SeqEmb(tokenized_sent)
            seq_gat_feature = self.GatEmb(amino_list, amino_degree_list, amino_mask)
        elif self.seq_model_type == "bert-nofinetune":
            self.SeqEmb.eval()
            with torch.no_grad():
                seq_bert_feature = self.SeqEmb(input_ids=tokenized_sent, attention_mask=attention_mask, token_type_ids=token_type_ids)[1]
            seq_gat_feature = self.GatEmb(amino_list, amino_degree_list, amino_mask)
        elif self.seq_model_type == "bert-finetune" or self.seq_model_type == "linear-aug-norm":
            seq_bert_feature = self.SeqEmb(input_ids=tokenized_sent, attention_mask=attention_mask, token_type_ids=token_type_ids)[1]
            seq_gat_feature = self.GatEmb(amino_list, amino_degree_list, amino_mask)
        elif self.seq_model_type == "only_seq":
            seq_bert_feature = self.SeqEmb(input_ids=tokenized_sent, attention_mask=attention_mask, token_type_ids=token_type_ids)[1]
        elif self.seq_model_type == "only_gat":
            seq_gat_feature = self.GatEmb(amino_list, amino_degree_list, amino_mask)
        elif self.seq_model_type == "bert-nopretrain":
            self.SeqEmb.eval()
            seq_bert_feature = self.SeqEmb(input_ids=tokenized_sent, attention_mask=attention_mask, token_type_ids=token_type_ids)[1]
            seq_gat_feature = self.GatEmb(amino_list, amino_degree_list, amino_mask)

        if self.seq_model_type == "only_seq": 
            seq_feature = seq_bert_feature
        elif self.seq_model_type == "only_gat":
            seq_feature = seq_gat_feature
        elif self.seq_model_type == "bilstm":
            seq_feature = seq_bert_feature
        elif self.seq_model_type == "linear-aug-norm":
            seq_bert_feature = self.seq_layernorm(seq_bert_feature)
            seq_gat_feature = self.gat_layernorm(seq_gat_feature)
            seq_feature = torch.cat([seq_bert_feature, seq_gat_feature], dim=1)
            seq_feature = self.predict_n(seq_feature)
        else:
            seq_feature = torch.cat([seq_bert_feature, seq_gat_feature], dim=1)
        
        return seq_feature

