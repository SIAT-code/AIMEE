# import ptvsd
# ptvsd.enable_attach(address = ('0.0.0.0', 5678))
# ptvsd.wait_for_attach()

import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoModel, AutoConfig, BertTokenizer, AutoModelWithLMHead
from torchviz import make_dot

import time
import numpy as np
from tqdm import trange
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
# from getFeatures import save_smiles_dicts, get_smiles_dicts, get_smiles_array, moltosvg_highlight
import gc
import sys
import pickle
import random
import csv
import json
import re
import argparse

# from tensorboardX import SummaryWriter

import copy
import pandas as pd
import scipy
# then import my own modules
from AttentiveFP import Fingerprint, Fingerprint_viz, save_smiles_dicts, get_smiles_dicts, get_smiles_array, \
    moltosvg_highlight
from network import Network
#无已有pickle时需全调用
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import QED
# get_ipython().run_line_magic('matplotlib', 'inline')
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
# from ProEmb.ProBert import model_bert, model_mlm
# from IPython.display import SVG, display

import seaborn as sns;

from torchsummary import summary


sns.set(color_codes=True)

class DataHandler():
    def __init__(self, raw_filename, max_atom_len=0, max_bond_len=0):
        def get_cm_dict(pickle_dir):
            seq_list = []
            cm_list = []
            id_list = []
            degree_list_dict = {}
            max_neighbor_num = 0

            for f in os.listdir(pickle_dir):
                f_degree = os.path.dirname(pickle_dir) + '/' + f.split('.')[0] + '_' + 'degree' + '_' + str(self.max_len) + '.pkl'
                f_cm = os.path.dirname(pickle_dir) + '/' + f.split('.')[0] + '_' + str(self.max_len) + '.pkl'
                if os.path.exists(f_degree) and os.path.exists(f_cm):
                    with open(f_degree, 'rb') as f_r:
                        degree_list_dict = pickle.load(f_r)
                    max_neighbor_num = list(degree_list_dict.items())[0][1].shape[1]
                    with open(f_cm, 'rb') as f_r:
                        cm_df = pickle.load(f_r)
                else:
                    f = os.path.join(pickle_dir, f)  # /home/eason/PPI/drug/GAT/data/3d_pdb/pdbbind_2016
                    data = pickle.load(open(f, 'rb'))  # PDB-ID seqs contact_map
                    for index, row in data.iterrows():
                        seq = row[cm_seq][:self.max_len]
                        if seq == '':
                            continue
                        cm = row['contact_map'][0][:self.max_len, :self.max_len]  # row['contact_map']:208×208

                        mn = np.max(np.sum(cm, axis=1))
                        if max_neighbor_num < mn:
                            max_neighbor_num = mn
                    
                    for index, row in data.iterrows():
                        seq = row[cm_seq][:self.max_len]
                        if seq == '':
                            continue
                        cm = row['contact_map'][0][:self.max_len, :self.max_len]  # row['contact_map']:208×208
                        cm_tmp = cm.astype(int)
                        cm_tmp = np.pad(cm_tmp, ((0, self.max_len - cm.shape[0]), (0, self.max_len - cm.shape[1])), 'constant', constant_values=(0, 0))
                        cm_list.append(cm_tmp)
                        seq_list.append(row[cm_seq])
                        id_list.append(row['PDB-ID'])

                        degree_list = []
                        for i in range(len(seq)):
                            tmp = np.array(np.where(cm[i] > 0.5)[0])
                            tmp = np.pad(tmp, (0, max_neighbor_num - tmp.shape[0]), 'constant', constant_values=(-1, -1))
                            degree_list.append(tmp)
                        
                        degree_list = np.stack(degree_list, 0)
                        degree_list = np.pad(degree_list, ((0, self.max_len - degree_list.shape[0]), (0, 0)), 'constant',
                                            constant_values=(-1, -1))

                        degree_list_dict[row['PDB-ID']] = degree_list
                    cm_df = pd.DataFrame({"PDB-ID": id_list, "seqs": seq_list, "cm_pad": cm_list})
                    with open(f_degree, 'wb') as f_w:
                        pickle.dump(degree_list_dict, f_w)
                    with open(f_cm, 'wb') as f_w:
                        pickle.dump(cm_df, f_w)
                    
            return degree_list_dict, max_neighbor_num, cm_df

        self.data_df, self.smile_feature_dict = self.load_smile(raw_filename, max_atom_len=max_atom_len, max_bond_len=max_bond_len)
        self.amino_dict = {}
        for key, value in vocab.items():
            if value - special_vocab_size >=  0:
                self.amino_dict[key] = value - special_vocab_size

        # for protein structure
        self.input_size = nonspecial_vocab_size

        self.max_len = max_seq_len   # 512
        self.enc_lib = np.eye(self.input_size)

        if model_type != "only_molecule":
            self.degree_list_dict, self.max_neighbor_num, self.cm_df = get_cm_dict(cm_pickle_dir)  # degree_list_dict:字典变量，每个氨基酸序列和对应的氨基酸之间的作用图（不是蛋白质之间的作用图）。

    def get_init(self, seq_list):
        mat = []
        for seq in seq_list:
            # seq = list(map(lambda ch: ord(ch) - ord('A'), seq[:self.max_len]))

            seq = [self.amino_dict[ch] for ch in seq[: self.max_len]]

            enc = self.enc_lib[seq]
            if enc.shape[0] < self.max_len:
                enc = np.pad(enc, ((0, self.max_len - enc.shape[0]), (0, 0)), 'constant')
            # print(enc.shape)

            mat.append(enc)
        mat = np.stack(mat, 0)

        mat = mat.astype(np.float32)

        return mat

    def get_degree_list(self, seq_list):
        mat = []
        for seq in seq_list:
            seq = seq[:self.max_len]
            if seq in self.degree_list_dict:
                cm = self.degree_list_dict[seq]
            else:
                # print('Sequence not found, ', seq)
                cm = np.ones([self.max_len, self.max_neighbor_num])
                cm = cm * -1
            mat.append(cm)
        mat = np.stack(mat, 0)

        return mat

    def get_amino_mask(self, seq_list):
        mat = []
        for seq in seq_list:
            mask = np.ones(min(len(seq), self.max_len), dtype=np.int)
            mask = np.pad(mask, (0, self.max_len - len(mask)), 'constant')
            mat.append(mask)
        mat = np.stack(mat, 0)
        # print('mask', mat)
        return mat

    def get_pro_structure(self, seq_list):

        # f1 = cal_mem()
        amino_list = self.get_init(seq_list)
        # f2 = cal_mem()
        # print('Get Pro Structure Index {}-{} costs: {}MB'.format('f2', 'f1', round(f1-f2, 4)))
        amino_degree_list = self.get_degree_list(seq_list)
        # f3 = cal_mem()
        # print('Get Pro Structure Index {}-{} costs: {}MB'.format('f2', 'f3', round(f2 - f3, 4)))
        amino_mask = self.get_amino_mask(seq_list)
        # f4 = cal_mem()
        # print('Get Pro Structure Index {}-{} costs: {}MB'.format('f3', 'f4', round(f3 - f4, 4)))

        return amino_list, amino_degree_list, amino_mask

    def load_smile(self, raw_filename, max_atom_len=0, max_bond_len=0):
        # raw_filename : "./PPI/drug/tasks/DTI/pdbbind/pafnucy_total_rdkit-smiles-v1.csv"
        filename = os.path.splitext(raw_filename)[0]
        ext_name = os.path.splitext(raw_filename)[-1]
        feature_filename = filename + '.pickle'
        prefix_filename = os.path.splitext(os.path.split(raw_filename)[-1])[0]
        # smiles_tasks_df : df : ["unnamed", "PDB-ID", "seq", "SMILES", "rdkit_smiles", "Affinity-Value", "set"]
        
        if ext_name == '.xlsx':
            smiles_tasks_df = pd.read_excel(io = raw_filename)  # main file
        elif ext_name == '.csv':
            smiles_tasks_df = pd.read_csv(raw_filename)  # main file
        else:
            sys.exit(1)
        # smilesList : array, 13464
        smilesList = smiles_tasks_df[SMILES].values
        print("number of all smiles: ", len(smilesList))
        atom_num_dist = []
        remained_smiles = []
        canonical_smiles_list = []
        for smiles in smilesList:
            try:
                mol = Chem.MolFromSmiles(smiles)  # input : smiles seqs, output : molecule obeject
                atom_num_dist.append(len(mol.GetAtoms()))  # list : get atoms obeject from molecule obeject
                remained_smiles.append(smiles)  # list : smiles without transformation error
                canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))  # canonical smiles without transformation error
            except:
                print("the smile is %s with transformation error" % smiles)
                pass
        print("number of successfully processed smiles after the first test: ", len(remained_smiles))

        "----------------------the first test----------------------"
        smiles_tasks_df = smiles_tasks_df[smiles_tasks_df[SMILES].isin(remained_smiles)]  # df(13464) : include smiles without transformation error
        # smiles_tasks_df[SMILES] = canonical_smiles_list
        smiles_tasks_df[SMILES] = remained_smiles

        smilesList = remained_smiles  # update valid smile

        # feature_dicts(dict) : 
        # {smiles_to_atom_info, smiles_to_atom_mask, smiles_to_atom_neighbors, "smiles_to_bond_info", "smiles_to_bond_neighbors", "smiles_to_rdkit_list"}
        if os.path.isfile(feature_filename):  # get smile feature dict
            feature_dicts = pickle.load(open(feature_filename, "rb"))
            print("load derectly!")
        else:
            feature_dicts = save_smiles_dicts(smilesList, filename, max_atom_len=max_atom_len, max_bond_len=max_bond_len)
            print("save pickle!")
        
        "----------------------the second test----------------------"
        remained_df = smiles_tasks_df[smiles_tasks_df[SMILES].isin(feature_dicts['smiles_to_atom_mask'].keys())]  # df(13435) : include smiles without transformation error and second test error
        # uncovered_index = ~smiles_tasks_df[SMILES].isin(feature_dicts['smiles_to_atom_mask'].keys())
        # uncovered_id = smiles_tasks_df["PDB-ID"][uncovered_index]
        # uncovered_df = smiles_tasks_df.drop(remained_df.index)
        print("number of successfully processed smiles after the second test: ", len(remained_df))

        return remained_df, feature_dicts


def tokenize(sent_list, vocab, seq_len):
    seq_len = seq_len + 2 # add [CLS] and [SEP]
    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []

    for sent in sent_list:
        attention_mask = [1 for _ in range(seq_len)]
        token_type_ids = [0 for _ in range(seq_len)]
        tmp = [vocab['[CLS]']]

        for word in sent:
            tmp.append(vocab[word])
            if len(tmp) == seq_len - 1:
                break
        tmp.append(vocab['[SEP]'])
        if len(tmp) < seq_len:
            for i in range(len(tmp), seq_len):
                tmp.append(vocab['[PAD]'])
                attention_mask[i] = 0

        all_input_ids.append(tmp)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)

    all_input_ids = np.array(all_input_ids)
    all_attention_mask = np.array(all_attention_mask)
    all_token_type_ids = np.array(all_token_type_ids)

    return torch.from_numpy(all_input_ids), torch.from_numpy(all_attention_mask), torch.from_numpy(all_token_type_ids)

# 如更改n-gram则此处要改
def create_sent(seq_list, seg_len=1):
    sent_list = []
    if seg_len == 1:
        for s in seq_list:
            sent_list.append(list(s))
    else:
        for s in seq_list:
            tmp = []
            for i in range(len(s) - seg_len + 1):
                tmp.append(s[i: i + seg_len])

            sent_list.append(tmp)
    return sent_list

def train(model, dataset, optimizer, loss_function, epoch):
    model.train()
    # np.random.seed(epoch)
    valList = list(dataset.index)
    np.random.shuffle(valList)
    batch_list = []

    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i + batch_size]
        batch_list.append(batch)

    for counter, batch in enumerate(batch_list):
        batch_df = dataset.loc[batch, :]
        smiles_list = batch_df[SMILES].values
        seq_list = batch_df.seqs.values

        y_val = batch_df[TASK].values

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,
                                                                                                     feature_dicts)
        amino_list, amino_degree_list, amino_mask = data_handler.get_pro_structure(seq_list)

        pro_seqs = batch_df.seqs.values

        sents = create_sent(pro_seqs)
        tokenized_sent, all_attention_mask, all_token_type_ids = tokenize(sents, vocab, max_seq_len)
        tokenized_sent = tokenized_sent.to(device)
        all_attention_mask = all_attention_mask.to(device)
        all_token_type_ids = all_token_type_ids.to(device)
        
        prediction = model(torch.Tensor(x_atom).to(device), torch.Tensor(x_bonds).to(device), torch.LongTensor(x_atom_index).to(device), \
                        torch.LongTensor(x_bond_index).to(device), torch.Tensor(x_mask).to(device), tokenized_sent, all_attention_mask, all_token_type_ids, \
                        torch.tensor(amino_list).to(device), torch.LongTensor(amino_degree_list).to(device), \
                        torch.Tensor(amino_mask).to(device))

        # loss = loss_function(prediction.view(-1, 2), torch.LongTensor(y_val).to(device).view(-1))
        # b = 0.9
        # flood = (loss - b).abs() + b
        true_labels = torch.LongTensor(y_val).to(device).view(-1)
        pred_labels = prediction.view(-1, 2)
        focal_loss = 0
        for true_label, pred_label in zip(true_labels, pred_labels):
            pred_label = pred_label - torch.max(pred_label)
            exp_pred_label = torch.exp(pred_label)
            softmax_pred_label = exp_pred_label / torch.sum(exp_pred_label)
            p = softmax_pred_label[true_label]
            focal_loss += -0.6 * (1-p)**2 * torch.log(p)

        optimizer.zero_grad()
        focal_loss.backward()
        optimizer.step()


    # writer.add_scalar('data/train_loss', np.mean(np.array(losses)).item(), epoch)


def evaluate(model, dataset, loss_function, fp_show=False):
    model.eval()
    # torch.no_grad()

    pred_list = []
    true_list = []
    # valList = np.arange(0, dataset.shape[0])
    valList = list(dataset.index)
    batch_list = []
    preds = None
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i + batch_size]
        batch_list.append(batch)
    for counter, batch in enumerate(batch_list):
        batch_df = dataset.loc[batch, :]
        smiles_list = batch_df[SMILES].values
        seq_list = batch_df.seqs.values
        #         print(batch_df)
        y_val = batch_df[TASK].values

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,
                                                                                                     feature_dicts)
        amino_list, amino_degree_list, amino_mask = data_handler.get_pro_structure(seq_list)

        pro_seqs = batch_df.seqs.values

        sents = create_sent(pro_seqs)
        tokenized_sent, all_attention_mask, all_token_type_ids = tokenize(sents, vocab, max_seq_len)
        tokenized_sent = tokenized_sent.to(device)
        all_attention_mask = all_attention_mask.to(device)
        all_token_type_ids = all_token_type_ids.to(device)
        with torch.no_grad():
            prediction = model(torch.Tensor(x_atom).to(device), torch.Tensor(x_bonds).to(device), torch.LongTensor(x_atom_index).to(device), \
                            torch.LongTensor(x_bond_index).to(device), torch.Tensor(x_mask).to(device), tokenized_sent, all_attention_mask, all_token_type_ids, \
                            torch.tensor(amino_list).to(device), torch.LongTensor(amino_degree_list).to(device), \
                            torch.Tensor(amino_mask).to(device))

        if preds is None:
            preds = prediction.detach().cpu().numpy()
        else:
            preds = np.append(preds, prediction.detach().cpu().numpy(), axis=0)

        true_list.extend(batch_df[TASK].values)
    
    loss = loss_function(torch.tensor(preds).to(device), torch.LongTensor(true_list).to(device).view(-1))
    pred_list = np.argmax(preds, axis=1)

    # auc_value = auc(pred_list, true_list)
    # auc_value = roc_auc_score(y_true=pred_list, y_score=true_list)
    f1 = f1_score(y_true=true_list, y_pred=pred_list)
    precision = precision_score(y_true=true_list, y_pred=pred_list)
    recall = recall_score(y_true=true_list, y_pred=pred_list)
    mcc = matthews_corrcoef(y_true=true_list, y_pred=pred_list)
    if fp_show:
        tn, fp, fn, tp = confusion_matrix(y_true=true_list, y_pred=pred_list).ravel()
        return loss, f1, precision, recall, mcc, fp
    else:
        return loss, f1, precision, recall, mcc

def predicted_value(model, dataset):
    model.eval()

    pred_list = []
    valList = list(dataset.index)
    batch_list = []
    preds = None
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i + batch_size]
        batch_list.append(batch)
    for counter, batch in enumerate(batch_list):
        batch_df = dataset.loc[batch, :]
        smiles_list = batch_df[SMILES].values

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,
                                                                                                     feature_dicts)

        if model_type != "only_molecule":
            seq_list = batch_df.seqs.values
            amino_list, amino_degree_list, amino_mask = data_handler.get_pro_structure(seq_list)
            pro_seqs = batch_df.seqs.values

            sents = create_sent(pro_seqs)
            tokenized_sent, all_attention_mask, all_token_type_ids = tokenize(sents, vocab, max_seq_len)
            tokenized_sent = tokenized_sent.to(device)
            all_attention_mask = all_attention_mask.to(device)
            all_token_type_ids = all_token_type_ids.to(device)
        
            with torch.no_grad():
                prediction = model(torch.Tensor(x_atom).to(device), torch.Tensor(x_bonds).to(device), torch.LongTensor(x_atom_index).to(device), \
                                torch.LongTensor(x_bond_index).to(device), torch.Tensor(x_mask).to(device), tokenized_sent, all_attention_mask, all_token_type_ids, \
                                torch.tensor(amino_list).to(device), torch.LongTensor(amino_degree_list).to(device), \
                                torch.Tensor(amino_mask).to(device))
        
        else:
            with torch.no_grad():
                prediction = model(torch.Tensor(x_atom).to(device), torch.Tensor(x_bonds).to(device), torch.LongTensor(x_atom_index).to(device), \
                                torch.LongTensor(x_bond_index).to(device), torch.Tensor(x_mask).to(device), None, None, None, None, None, None)
        
        if preds is None:
            preds = prediction.detach().cpu().numpy()
        else:
            preds = np.append(preds, prediction.detach().cpu().numpy(), axis=0)

        # pred_list = np.argmax(preds, axis=1)
    p_list = []
    for pred_label in preds:
        pred_label = torch.tensor(pred_label) - torch.max(torch.tensor(pred_label))
        exp_pred_label = torch.exp(pred_label)
        softmax_pred_label = exp_pred_label / torch.sum(exp_pred_label)
        p = softmax_pred_label[1]
        p_list.append(float(p))

    return p_list


def fun(radius, T, fingerprint_dim, weight_decay, learning_rate, p_dropout, pro_gat_dim, direction=False, load_model_path="", epochs=2, pre_param="", pre_model=None):
    loss_function = nn.CrossEntropyLoss()

    model = Network(int(round(radius)), int(round(T)), num_atom_features, num_bond_features,
                    int(round(fingerprint_dim)), p_dropout, pro_seq_dim, pro_seq_dim, pro_gat_dim, seq_model_type, task_type)

    with open("./generate_parameters.txt", 'w') as f:
        for param_name, param_value in model.named_parameters():
            print(param_name, ":", param_value.size(), file=f)
        print('Model parameters:', sum(param.numel() for param in model.parameters()), file=f)

    model = model.to(device)
    model = nn.DataParallel(model, device_ids=device_ids)

    if load_model_path:
        pre_state_dict = model.state_dict()
        print(list(pre_state_dict.items())[0])
        model.load_state_dict(torch.load(load_model_path, map_location="cpu"), strict=False)
        after_state_dict = model.state_dict()
        print(list(after_state_dict.items())[0])

    if pre_param == "":
        best_param = {}
        best_param["train_epoch"] = 0
        best_param["valid_epoch"] = 0
        best_param["train_loss"] = 9e8
        best_param["valid_loss"] = 9e8
    else:
        best_param = copy.deepcopy(pre_param)
        best_model = copy.deepcopy(pre_model)
        model = copy.deepcopy(pre_model)

    # Print model's state_dict
    print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    print('Model parameters:', sum(param.numel() for param in model.parameters()))

    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 10 ** -learning_rate, weight_decay=10 ** -weight_decay)
    optimizer = optim.Adam(model.parameters(), 10 ** -learning_rate, weight_decay=10 ** -weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience)

    plot_loss = []
    plot_precision = []
    plot_recall = []

    unchange_num = 0
    time_epochs = 0  # used to record real training epochs
    for epoch in range(epochs):
        start_time = time.time()
        train(model, train_df, optimizer, loss_function, epoch + 1)
        train_loss, train_f1, train_precision, train_recall, train_mcc, train_fp = evaluate(model, train_df, loss_function, fp_show=True)
        valid_loss, valid_f1, valid_precision, valid_recall, valid_mcc, valid_fp = evaluate(model, valid_df, loss_function, fp_show=True)
        test_loss, test_f1, test_precision, test_recall, test_mcc, test_fp = evaluate(model, test_df, loss_function, fp_show=True)

        scheduler.step(valid_f1)    # monitor mse and reduce lr

        # tensorboard add
        for para_group in optimizer.param_groups:
            lr = para_group['lr']
        # writer.add_scalar('data/learning_rates', lr, epoch)

        real_epoch = epoch+1
        if valid_loss < best_param["valid_loss"]:
            best_epoch = real_epoch
            best_param["train_loss"] = train_loss
            if pre_param == "":
                best_param["valid_epoch"] = real_epoch
            else:
                best_param["valid_epoch"] = real_epoch + pre_param["valid_epoch"]
            best_param["valid_loss"] = valid_loss
            best_model = copy.deepcopy(model)
            # torch.save(best_model.state_dict(), './PPI/drug/GAT/save/best-model-current.pth')
        end_time = time.time()
        
        train_log = 'epoch: {}, train_loss:{:.3f}, train_F1:{:.3f}, train_precision:{:.3f}, train_recall:{:.3f}, train_mcc:{:.3f}, train_fp:{}'.format(
            real_epoch, train_loss, train_f1, train_precision, train_recall, train_mcc, train_fp)
        valid_log = len('epoch: {}, '.format(epoch))*' '+'valid_loss:{:.3f}, valid_F1:{:.3f}, valid_precision:{:.3f}, valid_recall:{:.3f}, valid_mcc:{:.3f}, valid_fp:{}'.format(
            valid_loss, valid_f1, valid_precision, valid_recall, valid_mcc, valid_fp)
        test_log = len('epoch: {}, '.format(epoch))*' '+' test_loss:{:.3f},  test_F1:{:.3f},  test_precision:{:.3f},  test_recall:{:.3f},  test_mcc:{:.3f},  test_fp:{}, lr:{}'.format(
            test_loss, test_f1, test_precision, test_recall, test_mcc, test_fp, lr)
        each_epoch_time = "------------The {} epoch spend {}m-{:.4f}s------------".format(real_epoch, int((end_time-start_time)/60), 
              (end_time-start_time)%60)
        print(train_log)
        print(valid_log)
        print(test_log)
        print(each_epoch_time)
        with open(log_file, 'a') as f:
            f.write(train_log+'\n')
            f.write(valid_log+'\n')
            f.write(test_log+'\n')
            f.write(each_epoch_time+'\n')

        plot_loss.append([real_epoch, train_loss, valid_loss])
        plot_precision.append([real_epoch, train_precision, valid_precision])
        plot_recall.append([real_epoch, train_recall, valid_recall])

        time_epochs = time_epochs + 1
        if epoch != 0:
            if abs(last_valid_loss - valid_loss)/last_valid_loss <= 0.005 or valid_loss > last_valid_loss: 
                unchange_num = unchange_num+1
            else:
                unchange_num = 0
        if unchange_num == 10:  # second run don't stop early
            break
        last_valid_loss = valid_loss

    if pre_param == "":
        plot_loss = plot_loss[0: best_epoch]
        plot_precision = plot_precision[0: best_epoch]
        plot_recall = plot_recall[0: best_epoch]
        return plot_loss, plot_precision, plot_recall, best_param, best_model, time_epochs
    else:
        dir_save = "./save/" + model_type + "-{:.3f}-{}-{}-{}-cv{}".format(best_param['valid_loss'], best_param['valid_epoch'], pre_param["valid_epoch"]+real_epoch, nega_type, choose_cv)
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)
            print(dir_save+" create successful!")
        else:
            print(dir_save+" already exists.")
        os.system("cp " + log_file + ' ' + dir_save)
        torch.save(best_model.state_dict(), dir_save+'/best-model-{:.3f}-{}-{}.pth'.format(best_param['valid_loss'], best_param['valid_epoch'], pre_param["valid_epoch"]+real_epoch))
        print("radius:{}, T:{}, fingerprint_dim:{}, weight_decay:{}, learning_rate:{}, p_dropout:{}".format(radius, T, fingerprint_dim, weight_decay, learning_rate, p_dropout))
        return plot_loss, plot_precision, plot_recall, best_param, best_model, dir_save, time_epochs

def split_kfold(all_df):
    all_df.reset_index(drop=True, inplace=True)
    all_df['seqs'] = all_df.loc[0, 'seqs']

    positive_df = all_df[all_df[TASK].isin([1])]
    negative_df = all_df[all_df[TASK].isin([0])]

    positive_index_list = list(positive_df.index)
    negative_index_list = list(negative_df.index)
    random.shuffle(positive_index_list)
    random.shuffle(negative_index_list)

    per_posi_cv_num = int(len(positive_index_list)/cv_num)
    per_nega_cv_num = int(len(negative_index_list)/cv_num)
    cv_index_list = []
    for i in range(cv_num):
        if i == cv_num - 1:
            cv_index_list.append(positive_index_list[i*per_posi_cv_num:]+negative_index_list[i*per_nega_cv_num:])
        else:
            cv_index_list.append(positive_index_list[i*per_posi_cv_num: (i+1)*per_posi_cv_num]+negative_index_list[i*per_nega_cv_num: (i+1)*per_nega_cv_num])
    return cv_index_list


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True  #cpu/gpu结果一致
    torch.backends.cudnn.benchmark = False   #训练集变化不大时使训练加速

file_name = "./data/SARS-3CL_with_aff20_acsmpro_408pos_1358neg.xlsx"
test_file_name = "./data/3CL_enzymatic_activity_noaff_for_test_493pos_9459neg.xlsx"
new_add_file = "./data/SARS-3CL_414_1860_new_data_part.xlsx"

predict_file_name = "./data/mcule_201201_rdsmi_%d_small.csv"

nega_seed = 50  # [50, 3, 24]
parser = argparse.ArgumentParser()
parser.add_argument("--nega_seed", default=nega_seed, type=int, help="")
args = parser.parse_args()

nega_type = '8w_%d' % args.nega_seed
nega_file_name = "./data/SARS-3CL_no_aff_neg_test_29w_dropaff_8w_%d_small.xlsx" % args.nega_seed
load_model_path_list = ["./save/bert-finetune-0.016-102-147-8w_50-cv1/best-model-0.016-102-147.pth", \
                        "./save/bert-finetune-0.014-74-148-8w_50-cv2/best-model-0.014-74-148.pth", \
                        "./save/bert-finetune-0.012-148-168-8w_50-cv3/best-model-0.012-148-168.pth"]

model_type = "bert-finetune"
seq_model_type = model_type
task_type = "classification"
p_dropout = 0.4
epochs = 2
weight_decay = 4  # also known as l2_regularization_lambda
learning_rate = 3
patience = 30
radius = 3
T = 1
TASK = 'label'
SMILES = "rdkit_smiles"
cm_seq = 'seq'
nonspecial_vocab_size = 26
special_vocab_size = 5
max_seq_len = 512
fingerprint_dim=150
pro_gat_dim=64
n_gpu = 4
gpu_start = 0
cv_num = 3
per_gpu_batch_size = 32
do_train = False
do_predict = True
batch_size = per_gpu_batch_size * n_gpu
pro_seq_dim = 512
VOCAB_PATH = "./pretrained_model/protein_vocab.json"
cm_pickle_dir = './data/3d_pdb_v2'
seed = 3
set_seed(seed)
with open(VOCAB_PATH) as f:
    vocab = json.load(f)

if do_train:
    max_atom_len = 296 - 1
    max_bond_len = 304 - 1
    nega_data_handler = DataHandler(nega_file_name, max_atom_len=max_atom_len, max_bond_len=max_bond_len)
    nega_feature_dicts = nega_data_handler.smile_feature_dict
    nega_data_df = nega_data_handler.data_df
    nega_data_df_sample = nega_data_df

    data_handler = DataHandler(file_name, max_atom_len=max_atom_len, max_bond_len=max_bond_len)
    all_df = data_handler.data_df
    all_df = pd.concat([all_df, nega_data_df_sample], axis=0)
    cv_index_list = split_kfold(all_df)

    new_add_data_handler = DataHandler(new_add_file, max_atom_len=max_atom_len, max_bond_len=max_bond_len)
    new_add_feature_dicts = new_add_data_handler.smile_feature_dict
    new_add_data_df = new_add_data_handler.data_df
    new_add_cv_index_list = split_kfold(new_add_data_df)

    raw_feature_dicts = data_handler.smile_feature_dict
    test_data_handler = DataHandler(test_file_name, max_atom_len=max_atom_len, max_bond_len=max_bond_len)
    test_df = test_data_handler.data_df
    test_df['seqs'] = all_df.loc[0, 'seqs']
    test_feature_dicts = test_data_handler.smile_feature_dict

    feature_dicts = {'smiles_to_atom_mask':{}, 'smiles_to_atom_info':{}, 'smiles_to_bond_info':{},
                    'smiles_to_atom_neighbors':{}, 'smiles_to_bond_neighbors':{}, 'smiles_to_rdkit_list':{}}

    feature_dicts['smiles_to_atom_mask'].update(raw_feature_dicts['smiles_to_atom_mask'])
    feature_dicts['smiles_to_atom_mask'].update(test_feature_dicts['smiles_to_atom_mask'])
    feature_dicts['smiles_to_atom_mask'].update(nega_feature_dicts['smiles_to_atom_mask'])
    feature_dicts['smiles_to_atom_mask'].update(new_add_feature_dicts['smiles_to_atom_mask'])

    feature_dicts['smiles_to_atom_info'].update(raw_feature_dicts['smiles_to_atom_info'])
    feature_dicts['smiles_to_atom_info'].update(test_feature_dicts['smiles_to_atom_info'])
    feature_dicts['smiles_to_atom_info'].update(nega_feature_dicts['smiles_to_atom_info'])
    feature_dicts['smiles_to_atom_info'].update(new_add_feature_dicts['smiles_to_atom_info'])

    feature_dicts['smiles_to_bond_info'].update(raw_feature_dicts['smiles_to_bond_info'])
    feature_dicts['smiles_to_bond_info'].update(test_feature_dicts['smiles_to_bond_info'])
    feature_dicts['smiles_to_bond_info'].update(nega_feature_dicts['smiles_to_bond_info'])
    feature_dicts['smiles_to_bond_info'].update(new_add_feature_dicts['smiles_to_bond_info'])

    feature_dicts['smiles_to_atom_neighbors'].update(raw_feature_dicts['smiles_to_atom_neighbors'])
    feature_dicts['smiles_to_atom_neighbors'].update(test_feature_dicts['smiles_to_atom_neighbors'])
    feature_dicts['smiles_to_atom_neighbors'].update(nega_feature_dicts['smiles_to_atom_neighbors'])
    feature_dicts['smiles_to_atom_neighbors'].update(new_add_feature_dicts['smiles_to_atom_neighbors'])

    feature_dicts['smiles_to_bond_neighbors'].update(raw_feature_dicts['smiles_to_bond_neighbors'])
    feature_dicts['smiles_to_bond_neighbors'].update(test_feature_dicts['smiles_to_bond_neighbors'])
    feature_dicts['smiles_to_bond_neighbors'].update(nega_feature_dicts['smiles_to_bond_neighbors'])
    feature_dicts['smiles_to_bond_neighbors'].update(new_add_feature_dicts['smiles_to_bond_neighbors'])

    feature_dicts['smiles_to_rdkit_list'].update(raw_feature_dicts['smiles_to_rdkit_list'])
    feature_dicts['smiles_to_rdkit_list'].update(test_feature_dicts['smiles_to_rdkit_list'])
    feature_dicts['smiles_to_rdkit_list'].update(nega_feature_dicts['smiles_to_rdkit_list'])
    feature_dicts['smiles_to_rdkit_list'].update(new_add_feature_dicts['smiles_to_rdkit_list'])

    for choose_cv in range(1, cv_num+1):
        valid_df = all_df.iloc[cv_index_list[choose_cv-1], :]
        train_df = all_df.drop(cv_index_list[choose_cv-1], axis=0)

        valid_df_add = new_add_data_df.iloc[new_add_cv_index_list[choose_cv-1], :]
        train_df_add = new_add_data_df.drop(new_add_cv_index_list[choose_cv-1], axis=0)

        valid_df = pd.concat([valid_df, valid_df_add], axis=0)
        train_df = pd.concat([train_df, train_df_add], axis=0)

        valid_df.reset_index(drop=True, inplace=True)
        train_df.reset_index(drop=True, inplace=True)

        print("train_df_nums: %d, valid_df_nums: %d, test_df_nums: %d" % (len(train_df), len(valid_df), len(test_df)))

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = \
                            get_smiles_array([data_handler.data_df[SMILES][1]], feature_dicts)

        num_atom_features = x_atom.shape[-1]
        num_bond_features = x_bonds.shape[-1]

        prefix_filename = file_name.split('/')[-1].replace('.csv', '')
        start_time = str(time.ctime()).replace(':', '-').replace(' ', '_')
        log_file = './log/' + prefix_filename + '_1gram' + '_' + start_time + '.log'

        device = torch.device("cuda:{}".format(gpu_start) if torch.cuda.is_available() else "cpu")
        device_ids = []
        for i in range(n_gpu):
            device_ids.append(gpu_start+i)
        start_time_epochs = time.time()
        plot_loss1, plot_precision1, plot_recall1, best_param, best_model, time_epochs1 = fun(radius=radius, T=T, fingerprint_dim=fingerprint_dim, weight_decay=weight_decay, learning_rate=learning_rate, \
                                                   p_dropout=p_dropout, pro_gat_dim=pro_gat_dim, epochs=epochs)
        plot_loss2, plot_precision2, plot_recall2, best_param, best_model, dir_save, time_epochs2 = fun(radius=radius, T=T, fingerprint_dim=fingerprint_dim, weight_decay=weight_decay, learning_rate=learning_rate, p_dropout=p_dropout, pro_gat_dim=pro_gat_dim,
                                                        pre_param=best_param, pre_model=best_model)
        end_time_epochs = time.time()
        print("------------All epochs training spend {}h-{}m-{:.4f}s------------".format(
                    int((end_time_epochs-start_time_epochs)/3600), 
                    int((end_time_epochs-start_time_epochs)%3600/60), 
                    (end_time_epochs-start_time_epochs)%60))
if do_predict:
    for split_order in range(1, 2):
        predict_input_file = predict_file_name % split_order
        predict_output_file = os.path.splitext(predict_input_file)[0] + "_%s.csv"
        predict_data_handler = DataHandler(predict_input_file)
        data_handler = predict_data_handler
        feature_dicts = predict_data_handler.smile_feature_dict
        predict_data_df = predict_data_handler.data_df
        predict_data_df['seqs'] = "SGFRKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDVVYCPRHVICTSEDMLNPNYEDLLIRKSNHNFLVQAGNVQLRVIGHSMQNCVLKLKVDTANPKTPKYKFVRIQPGQTFSVLACYNGSPSGVYQCAMRPNFTIKGSFLNGSCGSVGFNIDYDCVSFCYMHHMELPTGVHAGTDLEGNFYGPFVDRQTAQAAGTDTTITVNVLAWLYAAVINGDRWFLNRFTTTLNDFNLVAMKYNYEPLTQDHVDILGPLSAQTGIAVLDMCASLKELLQNGMNGRTILGSALLEDEFTPFDVVRQCSGVTFQ"

        print("predict_df_nums: %d" % len(predict_data_df))

        for choose_cv in range(1, cv_num+1):
            output_save = "./save"
            file_name_list = os.listdir(output_save)
            for file_tmp in file_name_list:
                target_file_1 = re.findall(r'bert-finetune-.+-%s-cv%d' % (nega_type, choose_cv), file_tmp)
                if target_file_1 != []:
                    break
            file_name_list = os.listdir(os.path.join(output_save, target_file_1[0]))
            for file_tmp in file_name_list:
                target_file_2 = re.findall(r'.+pth', file_tmp)
                if target_file_2 != []:
                    break        
            load_model_path = os.path.join(output_save, target_file_1[0], target_file_2[0])

            x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = \
                                get_smiles_array([predict_data_handler.data_df[SMILES][1]], feature_dicts)

            num_atom_features = x_atom.shape[-1]
            num_bond_features = x_bonds.shape[-1]

            prefix_filename = file_name.split('/')[-1].replace('.csv', '')
            start_time = str(time.ctime()).replace(':', '-').replace(' ', '_')
            log_file = './log/' + prefix_filename + '_1gram' + '_' + start_time + '.log'

            device = torch.device("cuda:{}".format(gpu_start) if torch.cuda.is_available() else "cpu")
            device_ids = []
            for i in range(n_gpu):
                device_ids.append(gpu_start+i)
            model = Network(int(round(radius)), int(round(T)), num_atom_features, num_bond_features,
                            int(round(fingerprint_dim)), p_dropout, pro_seq_dim, pro_seq_dim, pro_gat_dim, seq_model_type, task_type)
            
            # model_dict = model.state_dict()
            # model_mlm_dict = model_mlm.state_dict()
            # model_mlm_dict =  {k: v for k, v in model_mlm_dict.items() if k in model_dict}
            # model_dict.update(model_mlm_dict)
            # model.load_state_dict(model_dict, strict=False)

            model = model.to(device)
            model = nn.DataParallel(model, device_ids=device_ids)

            for param_name, param_value in model.named_parameters():
                print(param_name, ":", param_value.size())

            model.load_state_dict(torch.load(load_model_path, map_location='cpu'), strict=False)
            prediction = predicted_value(model, predict_data_df)

            predict_data_df['cv'+str(choose_cv)] = prediction

            output_file = predict_output_file % nega_type
            print(output_file)
            ext_name = os.path.splitext(output_file)[-1]
            if ext_name == '.xlsx':
                predict_data_df.to_excel(output_file, index=False)
            elif ext_name == '.csv':
                predict_data_df.to_csv(output_file, index=False)
            else:
                sys.exit(-1)
        
