import torch
from bert_pytorch.model import BERT
from bert_pytorch.dataset.vocab import WordVocab
import csv
import numpy as np
import pandas as pd
import torch.nn as nn
import os
import pickle as pkl
from gensim.models.word2vec import Word2Vec
import torch.nn.functional as F

EMB_SIZE = 512

# MODEL_PATH = '/home/jian/protein/seq/bert_model_60w/bert.model.ep9'
# MODEL_PATH = '/home/fan/pycharm-temp/drug/seq/model_1gram_400w/bert.model.ep2'

# VOCAB_PATH = '/home/jian/protein/seq/vocab.small'
# VOCAB_PATH = '/home/fan/pycharm-temp/drug/seq/vocab-enz400w.big'
VOCAB_PATH = "./PPI/drug/seq/vocab-enz400w.big"
w2v_path = "./PPI/drug/seq/embedding.model"
voc_path = "./PPI/drug/seq/vocab_to_id.dict"
emb_path = "./PPI/drug/GAT/save/embeding_512d_10m.model"
vocab_path = "./PPI/drug/GAT/save/vocab_to_id_512d_10m.dict"

class BiLSTM(nn.Module):
    def __init__(self, vocab_size=28, char_embedding_size=EMB_SIZE, hidden_dims=512, 
                 num_classes=1, rnn_layers=1, keep_dropout=0.2):
        super(BiLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.char_embedding_size = char_embedding_size
        self.num_classes = num_classes
        self.keep_dropout = keep_dropout
        self.hidden_dims = hidden_dims
        self.rnn_layers = rnn_layers
        # 初始化字向量
        if os.path.exists(emb_path):
            embedding_pretrained = torch.tensor(pkl.load(open(emb_path, 'rb')).astype('float32'))
            self.char_embeddings = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
            print("using pretrained embedding:")
        else:
            self.char_embeddings = nn.Embedding(self.vocab_size, self.char_embedding_size)
            print("using newly-built embedding")
        # 字向量参与更新
        self.char_embeddings.weight.requires_grad = True
        # attention层
        self.attention_layer = nn.Sequential(
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True)
        )
        # 双层lstm
        self.lstm_net = nn.LSTM(self.char_embedding_size, self.hidden_dims,
                                num_layers=self.rnn_layers, dropout=self.keep_dropout,
                                bidirectional=True, batch_first=True)
        # FC层
        self.fc_out = nn.Sequential(
            nn.Dropout(self.keep_dropout),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, char_id):
        if not hasattr(self, '_flattened'):
            self.lstm_net.flatten_parameters()
            setattr(self, '_flattened', True)

        # input : [batch_size, len_seq, embedding_dim]
        sen_char_input = self.char_embeddings(char_id)
        # output : [batch_size, len_seq, n_hidden * 2]
        # output = output.permute(1, 0, 2)
        # final_hidden_state : [batch_size, num_layers * num_directions, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm_net(sen_char_input)
        final_hidden_state = final_hidden_state.permute(1, 0, 2)
        lstm_hidden = torch.mean(final_hidden_state, dim=1)
        final_output = self.fc_out(lstm_hidden)
        return final_output

class BiLSTM_Att(nn.Module):
    def __init__(self, vocab_size=28, char_embedding_size=EMB_SIZE, hidden_dims=512, 
                 num_classes=1, rnn_layers=1, keep_dropout=0.2):
        super(BiLSTM_Att, self).__init__()
        self.vocab_size = vocab_size
        self.char_embedding_size = char_embedding_size
        self.num_classes = num_classes
        self.keep_dropout = keep_dropout
        self.hidden_dims = hidden_dims
        self.rnn_layers = rnn_layers
        # 初始化字向量
        if os.path.exists(emb_path):
            embedding_pretrained = torch.tensor(pkl.load(open(emb_path, 'rb')).astype('float32'))
            self.char_embeddings = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
            print("using pretrained embedding:")
        else:
            self.char_embeddings = nn.Embedding(self.vocab_size, self.char_embedding_size)
            print("using newly-built embedding")
        # 字向量参与更新
        self.char_embeddings.weight.requires_grad = True
        # attention层
        self.attention_layer = nn.Sequential(
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True)
        )
        # 双层lstm
        self.lstm_net = nn.LSTM(self.char_embedding_size, self.hidden_dims,
                                num_layers=self.rnn_layers, dropout=self.keep_dropout,
                                bidirectional=True, batch_first=True)
        # FC层
        self.fc_out = nn.Sequential(
            nn.Dropout(self.keep_dropout),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True),
            # nn.Dropout(self.keep_dropout),
            # nn.Linear(self.hidden_dims, self.num_classes)
        )
        
    def attention_net_with_w(self, lstm_out, lstm_hidden):
        '''
        param lstm_out:    [batch_size, seq_len, hidden_dims * 2]
        param lstm_hidden: [batch_size, rnn_layers * num_directions, hidden_dims]
        return: [batch_size, hidden_dims]
        '''
        # 分块：lstm_tmp_out: [2, batch_size, seq_len, hidden_dims]
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        # h : [batch_size, seq_len, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        # lstm_hidden: [batch_size, hidden_dims]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        # lstm_hidden: [batch_size, 1, hidden_dims]
        # unsqueeze: 插入一维
        lstm_hidden = lstm_hidden.unsqueeze(1)
        # atten_w: [batch_size, 1, hidden_dims]
        atten_w = self.attention_layer(lstm_hidden)
        # m: [batch_size, seq_len, hidden_dims]
        m = nn.Tanh()(h)
        # atten_context: [batch_size, 1, seq_len]
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
        # softmax_w: [batch_size, 1, seq_len]
        softmax_w = F.softmax(atten_context, dim=-1)
        # context [batch_size, 1, hidden_dims]
        context = torch.bmm(softmax_w, h)
        # result: [batch_size, hidden_dims]
        # squeeze: 去掉一维
        result = context.squeeze(1)
        return result
    
    def forward(self, char_id):
        if not hasattr(self, '_flattened'):
            self.lstm_net.flatten_parameters()
            setattr(self, '_flattened', True)

        # input : [batch_size, len_seq, embedding_dim]
        sen_char_input = self.char_embeddings(char_id)
        # output : [batch_size, len_seq, n_hidden * 2]
        # output = output.permute(1, 0, 2)
        # final_hidden_state : [batch_size, num_layers * num_directions, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm_net(sen_char_input)
        final_hidden_state = final_hidden_state.permute(1, 0, 2)
        atten_out = self.attention_net_with_w(output, final_hidden_state)
        # lstm_hidden = torch.mean(final_hidden_state, dim=1)
        final_output = self.fc_out(atten_out)
        return final_output

def tokenize(sent_list, vocab, seq_len):
    t_seq_list = []

    for sent in sent_list:
        tmp = []
        for word in sent:
            tmp.append(vocab[word])
            if len(tmp) == seq_len:
                break
        if len(tmp) < seq_len:
            for i in range(seq_len - len(tmp)):
                tmp.append(vocab['<PAD>'])
        t_seq_list.append(tmp)

    t_seq_list = np.array(t_seq_list)

    return torch.from_numpy(t_seq_list)

# 如更改n-gram则此处要改
def create_sent(seq_list, seg_len=1):
    sent_list = []

    for s in seq_list:
        tmp = []
        for i in range(len(s) - seg_len + 1):
            tmp.append(s[i: i + seg_len])

        sent_list.append(tmp)
    return sent_list

def build_vocab(sentences, word_dim):
    if os.path.exists(w2v_path):
        # print("Loading Word2Vec model...")
        embedding = pkl.load(open(w2v_path, 'rb'))
        vocab_to_id = pkl.load(open(voc_path, 'rb'))
    else: 
        # print("Training Word2Vec model...")
        model = Word2Vec(sentences, size=word_dim)
        # 单词数
        print("vocab size:", len(model.wv.vocab.keys()))
        word_list = ['<PAD>', '<UNK>'] + list(model.wv.vocab.keys())
        vocab_to_id = {}
        for i in range(len(word_list)):
            vocab_to_id[word_list[i]] = i
        embedding = []
        # 加上PAD和UNK
        embedding.append([0 for i in range(word_dim)])
        embedding.append([0 for i in range(word_dim)])
        for k in model.wv.vocab.keys():
            embedding.append(model.wv[k])
        embedding = np.array(embedding) 
        pkl.dump(embedding, open(w2v_path, 'wb'))
        pkl.dump(vocab_to_id, open(voc_path, 'wb'))
    return vocab_to_id

def cal_pro_emb(seqs, model, device, seq_len):
    # vocab = WordVocab.load_vocab(VOCAB_PATH)  # ['<pad>', '<unk>', '<eos>', '<sos>', '<mask>', 'A', ...]
    sents = create_sent(seqs)  # seqs: ['YK...', 'ZQ...', ..., 'TR...']; sents: [[Y, K, ...],[Z, Q, ...],...,[T, R, ...]]
    vocab = build_vocab(sents, EMB_SIZE)
    tokenized_sent = tokenize(sents, vocab, seq_len)
    tokenized_sent = tokenized_sent.to(device)
    model.train()
    emb = model(tokenized_sent)
    return emb

def cal_pro_emb_eval(seqs, model, device, seq_len):
    sents = create_sent(seqs)
    vocab = build_vocab(sents, EMB_SIZE)
    tokenized_sent = tokenize(sents, vocab, seq_len)

    with torch.no_grad():
        model.eval()
        emb = model(tokenized_sent.to(device))
    return emb

#c
# if __name__ == '__main__':
#     df = pd.read_csv('/home/fan/pycharm-temp/drug/GAT/data/merge_data_pdbbind-v1_valid.csv')
#     seqs = df.pdb.values
#     print(seqs)
#     pro_embs = cal_pro_emb(seqs)
#     print(pro_embs)