import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from ProEmb.ProNetwork import ProNetwork
from AttentiveFP import Fingerprint


class Network(nn.Module):

    def __init__(self, radius, T, input_feature_dim, input_bond_dim, \
                 fingerprint_dim, p_dropout, pro_input_dim, pro_output_dim, pro_gat_dim, seq_model_type, task_type):
        super(Network, self).__init__()

        self.GAT = Fingerprint(radius, T, input_feature_dim, input_bond_dim, \
                               fingerprint_dim, p_dropout)

        if seq_model_type != "only_molecule":
            self.Pro = ProNetwork(pro_input_dim, pro_output_dim, pro_gat_dim, seq_model_type, radius, T, p_dropout=p_dropout)

        if seq_model_type == "only_gat":
            pro_output_dim = 0
        elif seq_model_type == "only_seq":
            pro_gat_dim = 0
        elif seq_model_type == "bilstm":
            pro_gat_dim = 0
        elif seq_model_type == "only_molecule":
            pro_output_dim = 0
            pro_gat_dim = 0

        self.seq_model_type = seq_model_type
        if task_type == "regression":
            self.predict_n = nn.Sequential(nn.Dropout(p_dropout),
                                        nn.Linear(fingerprint_dim + pro_output_dim + pro_gat_dim, 1))
        elif task_type == "classification":
            self.predict_n = nn.Sequential(nn.Dropout(p_dropout),
                                        nn.Linear(fingerprint_dim + pro_output_dim + pro_gat_dim, 2))
        
    def forward(self, atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask, tokenized_sent, attention_mask, token_type_ids, \
                amino_list, amino_degree_list, amino_mask):
        # pro_list B * Seq_len * word_size

        smile_feature = self.GAT(atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask)

        # pro_feature = self.Pro(pro_list)
        if self.seq_model_type == "only_molecule":
            con_feature = smile_feature
        else:
            pro_feature = self.Pro(amino_list, amino_degree_list, amino_mask, tokenized_sent, attention_mask, token_type_ids)

            con_feature = torch.cat((smile_feature, pro_feature), dim=1)

        prediction = self.predict_n(con_feature)

        return prediction

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Network(2, 1, 100, 100, 150, 0.5, 256, 256, 100).to(device)
    # model.forward (256, 178, 39) (256, 182, 10) (256, 178, 6) (256, 178, 6) (256, 178) (256, 256) (256, 512, 26) (256, 512, 25) (256, 512)
    summary(model, [(256, 178, 100), (256, 182, 10), (256, 178, 6),(256, 178, 6), (256, 178), (256, 256), (256, 512, 26) ,(256, 512, 25), (256, 512)])
