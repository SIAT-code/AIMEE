import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
import pandas as pd
import numpy as np


class ProGAT(nn.Module):

    def __init__(self, embedding_size, radius, T, p_dropout):
        super(ProGAT, self).__init__()

        '''def get_cm_dict(pickle_dir):

            cm_dict = {}

            for f in os.listdir(pickle_dir):
                f = pickle_dir + f 
                data = pickle.load(open(f, 'rb'))
                for index, row in data.iterrows():
                    seq = row['seqs'][:self.max_len]
                    cm = row['contact_map'][0][:self.max_len, :self.max_len]
                    cm_dict[seq] = cm
            return cm_dict

        self.enc_lib = np.eye(input_size)
        

        self.max_len = 10'''

        # pickle_dir = '/home/jian/protein/dataset/3d_pdbs/'

        # self.contact_map_dict = get_cm_dict(pickle_dir)
        input_size = 26

        self.emb_layer = nn.Linear(input_size, embedding_size)
        self.neighbor_fc = nn.Linear(input_size, embedding_size)
        self.GRUCell = nn.ModuleList([nn.GRUCell(embedding_size, embedding_size) for r in range(radius)])
        self.align = nn.ModuleList([nn.Linear(2 * embedding_size, 1) for r in range(radius)])
        self.attend = nn.ModuleList([nn.Linear(embedding_size, embedding_size) for r in range(radius)])

        self.seq_GRUCell = nn.GRUCell(embedding_size, embedding_size)
        self.seq_align = nn.Linear(2 * embedding_size, 1)
        self.seq_attend = nn.Linear(embedding_size, embedding_size)

        self.dropout = nn.Dropout(p=p_dropout)

        self.radius = radius
        self.T = T

    '''def get_init(self, seq_list):

        mat = []
        for seq in seq_list:
            seq = list(map(lambda ch : ord(ch) - ord('A'), seq[:self.max_len]))

            enc = self.enc_lib[seq]
            if enc.shape[0] < self.max_len:
                enc = np.pad(enc,((0, self.max_len - enc.shape[0]), (0, 0)), 'constant')
            #print(enc.shape)

            mat.append(enc)
        mat = np.stack(mat, 0)

        return torch.tensor(mat, dtype=torch.float32).cuda()

    def get_degree_list(self, seq_list):

        mat = []
        for seq in seq_list:
            seq = seq[:self.max_len]
            if seq in self.contact_map_dict:
                cm = self.contact_map_dict[seq]
            else:
                cm = np.random.rand(self.max_len, self.max_len)

            degree_list = []
            for i in range(len(seq)):
                tmp = np.array(np.where(cm[i] > 0.5)[0])
                tmp = np.pad(tmp, (0, self.max_len - tmp.shape[0]), 'constant', constant_values=(-1,-1))
                degree_list.append(tmp)

            degree_list = np.stack(degree_list, 0)
            degree_list = np.pad(degree_list, ((0, self.max_len - degree_list.shape[0]), (0, 0)), 'constant', constant_values=(-1, -1))
            mat.append(degree_list)
        mat = np.stack(mat, 0)
        
        return torch.tensor(mat)

    def get_amino_mask(self, seq_list):
        mat = []
        for seq in seq_list:

            mask = np.ones(min(len(seq), self.max_len), dtype=np.int)
            mask = np.pad(mask, (0, self.max_len - len(mask)), 'constant')
            mat.append(mask)
        mat = np.stack(mat, 0)
        #print('mask', mat)
        return torch.tensor(mat, dtype=torch.float32).cuda()'''

    def forward(self, amino_list, amino_degree_list, amino_mask):
        '''
        amino_list is (batch, max_seqlen(512), amino_nums(26)). One mat(512, 26) is a one-hot representation of seq based on atom feature numbers.
        amino_degree_list is (batch, max_seqlen(512), max_neighbors(21?)). One mat(512, 21) is the index of the amino's neighbors, -1 is padding.
        amino_mask is (batch, 512), the existed is 1 while padding is 0.
        '''
        # amino_list = self.get_init(seq_list)
        # amino_degree_list = self.get_degree_list(seq_list)
        # amino_mask = self.get_amino_mask(seq_list)

        amino_mask = amino_mask.unsqueeze(2)

        batch_size, seq_len, num_amino_feature = amino_list.shape

        # print(amino_list.shape)
        # (512, 26) -> (512, 64)  embed amino num_amino_feature
        amino_feature = F.leaky_relu(self.emb_layer(amino_list))

        # (batch, max_len, max_neighbors, amino_nums)
        neighbor = [amino_list[i][amino_degree_list[i]] for i in range(batch_size)]
        neighbor = torch.stack(neighbor, dim=0)

        # (batch, 512, 21, 26) -> (batch, 512, 21, 64)
        neighbor_feature = F.leaky_relu(self.neighbor_fc(neighbor))

        # generate mask to eliminate the influence of blank atoms
        attend_mask = amino_degree_list.clone()
        attend_mask[attend_mask != -1] = 1
        attend_mask[attend_mask == -1] = 0
        attend_mask = attend_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        softmax_mask = amino_degree_list.clone()
        softmax_mask[softmax_mask != -1] = 0
        softmax_mask[softmax_mask == -1] = -9e8  # make the softmax value extremly small
        softmax_mask = softmax_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        batch_size, seq_length, max_neighbor_num, embedding_dim = neighbor_feature.shape
        amino_feature_expand = amino_feature.unsqueeze(-2).expand(batch_size, seq_length, max_neighbor_num,
                                                                  embedding_dim)
        feature_align = torch.cat([amino_feature_expand, neighbor_feature], dim=-1)

        # align[0] relate to radius， align_score -> (batch, max_len, max_neighbor, 1)
        align_score = F.leaky_relu(self.align[0](self.dropout(feature_align)))

        align_score = align_score + softmax_mask
        attention_weight = F.softmax(align_score, -2)
        #             print(attention_weight)
        attention_weight = attention_weight * attend_mask
        #         print(attention_weight)
        neighbor_feature_transform = self.attend[0](self.dropout(neighbor_feature))
        #             print(features_neighbor_transform.shape)
        # 邻节点每维特征*attention权重(21, 64)，然后所有ci相加得到当前节点c(1,64)
        context = torch.sum(torch.mul(attention_weight, neighbor_feature_transform), -2)
        context = F.elu(context)
        # attention_weight -> torch.Size([batch, 512, max_neighbor, 1])
        # context -> torch.Size([batch, 512, 64])

        context_reshape = context.view(batch_size * seq_length, embedding_dim)
        amino_feature_reshape = amino_feature.view(batch_size * seq_length, embedding_dim)
        amino_feature_reshape = self.GRUCell[0](context_reshape, amino_feature_reshape)
        amino_feature = amino_feature_reshape.view(batch_size, seq_length, embedding_dim)

        # do nonlinearity
        activated_features = F.relu(amino_feature)

        for d in range(self.radius - 1):
            # bonds_indexed = [bond_list[i][torch.cuda.LongTensor(bond_degree_list)[i]] for i in range(batch_size)]
            neighbor_feature = [activated_features[i][amino_degree_list[i]] for i in range(batch_size)]

            # neighbor_feature is a list of 3D tensor, so we need to stack them into a 4D tensor first
            neighbor_feature = torch.stack(neighbor_feature, dim=0)
            amino_feature_expand = activated_features.unsqueeze(-2).expand(batch_size, seq_length, max_neighbor_num,
                                                                           embedding_dim)

            feature_align = torch.cat([amino_feature_expand, neighbor_feature], dim=-1)

            align_score = F.leaky_relu(self.align[d + 1](self.dropout(feature_align)))
            #             print(attention_weight)
            align_score = align_score + softmax_mask
            attention_weight = F.softmax(align_score, -2)
            #             print(attention_weight)
            attention_weight = attention_weight * attend_mask
            #             print(attention_weight)
            neighbor_feature_transform = self.attend[d + 1](self.dropout(neighbor_feature))
            #             print(features_neighbor_transform.shape)
            context = torch.sum(torch.mul(attention_weight, neighbor_feature_transform), -2)
            #             print(context.shape)
            context = F.elu(context)
            context_reshape = context.view(batch_size * seq_length, embedding_dim)
            #             atom_feature_reshape = atom_feature.view(batch_size*mol_length, fingerprint_dim)
            amino_feature_reshape = self.GRUCell[d + 1](context_reshape, amino_feature_reshape)
            amino_feature = amino_feature_reshape.view(batch_size, seq_length, embedding_dim)

            # do nonlinearity
            activated_features = F.relu(amino_feature)
            # print('act', activated_features.shape)

        # 将aa特征汇聚成蛋白序列的
        seq_feature = torch.sum(activated_features * amino_mask, dim=-2)  # B * feature, all atom sum = mol

        # do nonlinearity
        activated_features_mol = F.relu(seq_feature)

        seq_softmax_mask = amino_mask.clone()
        seq_softmax_mask[seq_softmax_mask == 0] = -9e8
        seq_softmax_mask[seq_softmax_mask == 1] = 0
        seq_softmax_mask = seq_softmax_mask.type(torch.cuda.FloatTensor)

        for t in range(self.T):
            seq_prediction_expand = activated_features_mol.unsqueeze(-2).expand(batch_size, seq_length, embedding_dim)
            seq_align = torch.cat([seq_prediction_expand, activated_features], dim=-1)
            seq_align_score = F.leaky_relu(self.seq_align(seq_align))
            seq_align_score = seq_align_score + seq_softmax_mask
            seq_attention_weight = F.softmax(seq_align_score, -2)
            seq_attention_weight = seq_attention_weight * amino_mask
            #             print(mol_attention_weight.shape,mol_attention_weight)
            activated_features_transform = self.seq_attend(self.dropout(activated_features))
            #             aggregate embeddings of atoms in a molecule
            seq_context = torch.sum(torch.mul(seq_attention_weight, activated_features_transform), -2)
            #             print(mol_context.shape,mol_context)
            seq_context = F.elu(seq_context)
            seq_feature = self.seq_GRUCell(seq_context, seq_feature)
            #             print(mol_feature.shape,mol_feature)

            # do nonlinearity
            activated_features_seq = F.relu(seq_feature)

            # print(activated_features.shape)

        return activated_features_seq

        # neighbor = [amino_input[i] for i in range(batch_size)]

