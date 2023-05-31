import json
import os
import random
import sys

import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
import numpy as np
from tools.utils import tokenize
from torchtext.data import Field
from torchtext.vocab import Vectors
curPath = os.path.abspath(os.path.dirname('__file__'))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

class MashupAPI(Dataset):
    def __init__(self, mashup_api_file):
        super().__init__()
        with open(rootPath + '/data/mashup_description.json', 'r') as f:
            self.mashup_description = json.load(f)
        with open(rootPath + '/data/used_api_description.json', 'r') as f:
            self.api_description = json.load(f)
        with open(rootPath + '/data/'+mashup_api_file, 'r') as f:
            self.mashup_api = json.load(f)
        self.length=len(self.mashup_api)
        self.api_num=len(self.api_description)
        self.mashup_num=len(self.mashup_description)
        self.embed=np.eye(self.api_num)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        mashup_api = self.mashup_api[index]
        used_api=[]
        mashup_index = mashup_api[0]
        api_index=mashup_api[1]

        m_desc = self.mashup_description[mashup_api[0]]
        api_desc = self.api_description[mashup_api[1]]
        used_api.append(mashup_api[1])
        des_len = len(m_desc) if len(m_desc) < 50 else 50
        api_len = len(api_desc) if len(api_desc) < 50 else 50

        rand_offset = random.randint(0, self.api_num - 1)
        api_neg_desc = self.api_description[rand_offset]
        api_nag_len = len(api_neg_desc) if len(api_neg_desc) < 50 else 50

        used_api_numpy=self.embed[used_api]
        used_api_tensor = torch.tensor(used_api_numpy, dtype=torch.float).squeeze()

        m_desc = torch.tensor(m_desc).long()
        api_desc = torch.tensor(api_desc).long()
        api_neg_desc = torch.tensor(api_neg_desc).long()
        #api_neg_desc, api_nag_len,
        return m_desc, des_len, api_desc, api_len,  mashup_index,api_index,used_api_tensor

class MyDataset:
    def __init__(self):
        cache = '.vec_cache'
        if not os.path.exists(cache):
            os.mkdir(cache)
        self.mashup_ds = MashupAPI('train_mashup_api.json')
        self.test_mashup_ds = MashupAPI('test_mashup_api.json')
        self.max_vocab_size = 10000
        self.max_doc_len = 50
        self.vectors = Vectors(name=rootPath + '\glove.6B.300d.txt', cache=cache)
        self.field = Field(sequential=True, tokenize=tokenize, lower=True, fix_length=self.max_doc_len)
        self.field.build_vocab(self.mashup_ds.mashup_description, self.mashup_ds.api_description, vectors=self.vectors, min_freq=1, max_size=self.max_vocab_size)
        self.random_seed = 2020
        self.num_mashup = self.mashup_ds.mashup_num
        self.num_api = self.mashup_ds.api_num
        self.vocab_size = len(self.field.vocab)
        self.embed = self.field.vocab.vectors
        self.embed_dim = self.vectors.dim
        #print("embed:",len(self.embed),"embed_dim:",self.embed_dim,"vocab_size:",self.vocab_size)
        self.des_lens = []
        self.word2id()

    def word2id(self):
        for i, des in enumerate(self.mashup_ds.mashup_description):
            tokens = [self.field.vocab.stoi[x] for x in des]
            if not tokens:
                tokens = [0]
            if len(tokens) < self.max_doc_len:
                tokens.extend([1] * (self.max_doc_len - len(tokens)))
            else:
                tokens = tokens[:self.max_doc_len]
            self.mashup_ds.mashup_description[i] = tokens

        for i, des in enumerate(self.mashup_ds.api_description):
            tokens = [self.field.vocab.stoi[x] for x in des]
            if not tokens:
                tokens = [0]
            if len(tokens) < self.max_doc_len:
                tokens.extend([1] * (self.max_doc_len - len(tokens)))
            else:
                tokens = tokens[:self.max_doc_len]
            self.mashup_ds.api_description[i] = tokens

        for i, des in enumerate(self.test_mashup_ds.mashup_description):
            tokens = [self.field.vocab.stoi[x] for x in des]
            if not tokens:
                tokens = [0]
            if len(tokens) < self.max_doc_len:
                tokens.extend([1] * (self.max_doc_len - len(tokens)))
            else:
                tokens = tokens[:self.max_doc_len]
            self.test_mashup_ds.mashup_description[i] = tokens

        for i, des in enumerate(self.test_mashup_ds.api_description):
            tokens = [self.field.vocab.stoi[x] for x in des]
            if not tokens:
                tokens = [0]
            if len(tokens) < self.max_doc_len:
                tokens.extend([1] * (self.max_doc_len - len(tokens)))
            else:
                tokens = tokens[:self.max_doc_len]
            self.test_mashup_ds.api_description[i] = tokens

class FCDataset:
    def __init__(self):
        cache = '.vec_cache'
        if not os.path.exists(cache):
            os.mkdir(cache)
        self.mashup_ds = FCMashupAPI('train_mashup_api.json',is_training=True)
        self.test_mashup_ds = FCMashupAPI('test_mashup_api.json',is_training=False)
        self.max_vocab_size = 10000
        self.max_doc_len = 50
        self.vectors = Vectors(name=rootPath + '\glove.6B.300d.txt', cache=cache)
        self.field = Field(sequential=True, tokenize=tokenize, lower=True, fix_length=self.max_doc_len)
        self.field.build_vocab(self.mashup_ds.mashup_description, self.mashup_ds.api_description, vectors=self.vectors, min_freq=1, max_size=self.max_vocab_size)
        self.random_seed = 2020
        self.num_mashup = self.mashup_ds.mashup_num
        self.num_api = self.mashup_ds.api_num
        with open(rootPath + '/data/category_list.json', 'r') as f:
            category_list = json.load(f)
        self.num_category = len(category_list)
        self.vocab_size = len(self.field.vocab)
        self.embed = self.field.vocab.vectors
        self.embed_dim = self.vectors.dim
        #print("embed:",len(self.embed),"embed_dim:",self.embed_dim,"vocab_size:",self.vocab_size)
        self.des_lens = []
        self.word2id()
        self.tag2feature()

    def word2id(self):
        for i, des in enumerate(self.mashup_ds.mashup_description):
            tokens = [self.field.vocab.stoi[x] for x in des]
            if not tokens:
                tokens = [0]
            if len(tokens) < self.max_doc_len:
                tokens.extend([1] * (self.max_doc_len - len(tokens)))
            else:
                tokens = tokens[:self.max_doc_len]
            self.mashup_ds.mashup_description[i] = tokens

        for i, des in enumerate(self.mashup_ds.api_description):
            tokens = [self.field.vocab.stoi[x] for x in des]
            if not tokens:
                tokens = [0]
            if len(tokens) < self.max_doc_len:
                tokens.extend([1] * (self.max_doc_len - len(tokens)))
            else:
                tokens = tokens[:self.max_doc_len]
            self.mashup_ds.api_description[i] = tokens

        for i, des in enumerate(self.test_mashup_ds.mashup_description):
            tokens = [self.field.vocab.stoi[x] for x in des]
            if not tokens:
                tokens = [0]
            if len(tokens) < self.max_doc_len:
                tokens.extend([1] * (self.max_doc_len - len(tokens)))
            else:
                tokens = tokens[:self.max_doc_len]
            self.test_mashup_ds.mashup_description[i] = tokens

        for i, des in enumerate(self.test_mashup_ds.api_description):
            tokens = [self.field.vocab.stoi[x] for x in des]
            if not tokens:
                tokens = [0]
            if len(tokens) < self.max_doc_len:
                tokens.extend([1] * (self.max_doc_len - len(tokens)))
            else:
                tokens = tokens[:self.max_doc_len]
            self.test_mashup_ds.api_description[i] = tokens

    def tag2feature(self):
        for i, category in enumerate(self.mashup_ds.mashup_category):
            tokens = [self.field.vocab.stoi[x] for x in tokenize(' '.join(category))]
            if not tokens:
                tokens = [0]
            if len(tokens) < 10:
                tokens.extend([1] * (10 - len(tokens)))
            else:
                tokens = tokens[:10]
            self.mashup_ds.mashup_category_token.append(tokens)
        for i, category in enumerate(self.mashup_ds.api_category):
            tokens = [self.field.vocab.stoi[x] for x in tokenize(' '.join(category))]
            if not tokens:
                tokens = [0]
            if len(tokens) < 10:
                tokens.extend([1] * (10 - len(tokens)))
            else:
                tokens = tokens[:10]
            self.mashup_ds.api_category_token.append(tokens)
        for i, category in enumerate(self.test_mashup_ds.mashup_category):
            tokens = [self.field.vocab.stoi[x] for x in tokenize(' '.join(category))]
            if not tokens:
                tokens = [0]
            if len(tokens) < 10:
                tokens.extend([1] * (10 - len(tokens)))
            else:
                tokens = tokens[:10]
            self.test_mashup_ds.mashup_category_token.append(tokens)
        for i, category in enumerate(self.test_mashup_ds.api_category):
            tokens = [self.field.vocab.stoi[x] for x in tokenize(' '.join(category))]
            if not tokens:
                tokens = [0]
            if len(tokens) < 10:
                tokens.extend([1] * (10 - len(tokens)))
            else:
                tokens = tokens[:10]
            self.test_mashup_ds.api_category_token.append(tokens)

class FCMashupAPI(Dataset):
    def __init__(self, mashup_api_file,is_training):
        super(FCMashupAPI, self).__init__()
        with open(rootPath + '/data/mashup_description.json', 'r') as f:
            self.mashup_description = json.load(f)
        with open(rootPath + '/data/used_api_description.json', 'r') as f:
            self.api_description = json.load(f)
        with open(rootPath + '/data/'+mashup_api_file, 'r') as f:
            self.mashup_api = json.load(f)
        with open(rootPath + '/data/mashup_category.json', 'r') as f:
            self.mashup_category = json.load(f)
        with open(rootPath + '/data/used_api_category.json', 'r') as f:
            self.api_category = json.load(f)
        with open(rootPath + '/data/mashup_name.json', 'r') as f:
            self.mashup_name = json.load(f)
        with open(rootPath + '/data/used_api_list.json', 'r') as f:
            self.api_name = json.load(f)

        self.length=len(self.mashup_api)
        self.api_num=len(self.api_description)
        self.mashup_num=len(self.mashup_description)

        self.mashup_category_token = []
        self.api_category_token = []
        self.triplet = []
        if is_training:
            pos_mashup_api = self.mashup_api
            self.neg_num = 14  # 一个正例对应需要采样的负例数量
            for indice in pos_mashup_api:
                mashup_idx = indice[0]
                pos_api_idx = indice[1]
                self.triplet.append([mashup_idx, pos_api_idx, 1])
                for idx in range(self.neg_num):
                    rand_offset = random.randint(0, self.api_num - 1)
                    if rand_offset != pos_api_idx:
                        self.triplet.append([mashup_idx, rand_offset, -1])
        else:
            pos_mashup_api = self.mashup_api
            for indice in pos_mashup_api:
                mashup_idx = indice[0]
                pos_api_idx = indice[1]
                self.triplet.append([mashup_idx, pos_api_idx, 1])

    def __len__(self):
        return len(self.triplet)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample = self.triplet[index]
        mashup_idx = sample[0]
        api_idx = sample[1]

        m_desc = self.mashup_description[mashup_idx]
        des_len = len(m_desc) if len(m_desc) < 50 else 50
        mashup_category_token = torch.LongTensor(self.mashup_category_token[mashup_idx])

        api_desc = self.api_description[api_idx]
        api_len = len(api_desc) if len(api_desc) < 50 else 50
        api_category_token = torch.LongTensor(self.api_category_token[api_idx])

        m_desc = torch.tensor(m_desc).long()
        api_desc = torch.tensor(api_desc).long()
        label = sample[2]
        return m_desc, des_len, mashup_category_token, api_desc, api_len, api_category_token, label

class MTFMDataset:
    def __init__(self):
        cache = '.vec_cache'
        if not os.path.exists(cache):
            os.mkdir(cache)
        self.mashup_ds = M_Dataset('train_mashup_api.json')
        self.test_mashup_ds = M_Dataset('test_mashup_api.json')
        self.max_vocab_size = 10000
        self.max_doc_len = 50
        self.vectors = Vectors(name=rootPath + '\glove.6B.300d.txt', cache=cache)
        self.field = Field(sequential=True, tokenize=tokenize, lower=True, fix_length=self.max_doc_len)
        self.field.build_vocab(self.mashup_ds.mashup_description, self.mashup_ds.api_description, vectors=self.vectors, min_freq=1, max_size=self.max_vocab_size)
        self.random_seed = 2020
        self.num_category = self.mashup_ds.num_category
        self.num_mashup = self.mashup_ds.mashup_num
        self.num_api = self.mashup_ds.api_num
        self.vocab_size = len(self.field.vocab)
        self.embed = self.field.vocab.vectors
        self.embed_dim = self.vectors.dim
        self.des_lens = []
        self.word2id()
        self.tag2feature()

    def word2id(self):
        for i, des in enumerate(self.mashup_ds.mashup_description):
            tokens = [self.field.vocab.stoi[x] for x in des]
            if not tokens:
                tokens = [0]
            if len(tokens) < self.max_doc_len:
                tokens.extend([1] * (self.max_doc_len - len(tokens)))
            else:
                tokens = tokens[:self.max_doc_len]
            self.mashup_ds.mashup_description[i] = tokens


        for i, des in enumerate(self.test_mashup_ds.mashup_description):
            tokens = [self.field.vocab.stoi[x] for x in des]
            if not tokens:
                tokens = [0]
            if len(tokens) < self.max_doc_len:
                tokens.extend([1] * (self.max_doc_len - len(tokens)))
            else:
                tokens = tokens[:self.max_doc_len]
            self.test_mashup_ds.mashup_description[i] = tokens

    def tag2feature(self):
        for i, category in enumerate(self.mashup_ds.category):
            tokens = [self.field.vocab.stoi[x] for x in tokenize(' '.join(category))]
            if not tokens:
                tokens = [0]
            if len(tokens) < 10:
                tokens.extend([1] * (10 - len(tokens)))
            else:
                tokens = tokens[:10]
            self.mashup_ds.category_token.append(tokens)

class M_Dataset(Dataset):
    def __init__(self, mashup_api_file):
        super().__init__()
        with open(rootPath + '/data/mashup_description.json', 'r') as f:
            self.mashup_description = json.load(f)
        with open(rootPath + '/data/used_api_description.json', 'r') as f:
            self.api_description = json.load(f)
        with open(rootPath + '/data/mashup_category.json', 'r') as f:
            self.category = json.load(f)
        with open(rootPath + '/data/' + mashup_api_file, 'r') as f:
            self.mashup_api = json.load(f)
        with open(rootPath + '/data/category_list.json', 'r') as f:
            category_list = json.load(f)
        with open(rootPath + '/data/mashup_used_api.json', 'r') as f:
            self.used_api = json.load(f)

        with open(rootPath + '/data/used_api_list.json', 'r') as f:
            api_list = json.load(f)
        self.used_api_mlb = MultiLabelBinarizer()
        self.used_api_mlb.fit([api_list])
        self.api_num = len(self.api_description)
        self.mashup_num = len(self.mashup_description)
        self.num_category = len(category_list)
        self.category_mlb = MultiLabelBinarizer()
        self.category_mlb.fit([category_list])
        self.length = len(self.mashup_api)
        self.embed = np.eye(self.api_num)

        self.des_lens = []
        self.category_token = []
        for des in self.mashup_description:
            self.des_lens.append(len(des) if len(des) < 50 else 50)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        mashup_api = self.mashup_api[index]
        used_api = []
        mashup_index = mashup_api[0]
        api_index = mashup_api[1]

        description = self.mashup_description[mashup_index]
        category_tensor = torch.tensor(self.category_mlb.transform([self.category[mashup_index]]), dtype=torch.long).squeeze()
        used_api.append(api_index)
        used_api_numpy = self.embed[used_api]
        used_api_tensor = torch.tensor(used_api_numpy, dtype=torch.float).squeeze()
        des_len = torch.tensor(self.des_lens[mashup_index])
        #category_token = torch.LongTensor(self.category_token[mashup_index])
        return  torch.tensor(description).long(), used_api_tensor, category_tensor,des_len,torch.tensor(mashup_index).long()