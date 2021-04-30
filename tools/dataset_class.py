from torch.utils.data import Dataset
import sys
import os
import json
curPath = os.path.abspath(os.path.dirname('__file__'))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import torch
from src.utils import tokenize
from torchtext.data import Field
from torchtext.vocab import Vectors
from random import randint, choice
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np


class MashupDataset(Dataset):
    def __init__(self, all_api=False):
        super().__init__()
        with open(rootPath + '/data/mashup_name.json', 'r') as f:
            self.name = json.load(f)
        with open(rootPath + '/data/mashup_description.json', 'r') as f:
            self.description = json.load(f)
        with open(rootPath + '/data/mashup_category.json', 'r') as f:
            self.category = json.load(f)
        with open(rootPath + '/data/mashup_used_api.json', 'r') as f:
            self.used_api = json.load(f)
        with open(rootPath + '/data/category_list.json', 'r') as f:
            category_list = json.load(f)

        if all_api:
            with open(rootPath + '/data/api_name.json', 'r') as f:
                api_list = json.load(f)
        else:
            with open(rootPath + '/data/used_api_list.json', 'r') as f:
                api_list = json.load(f)
        self.num_api = len(api_list)
        self.num_category = len(category_list)
        self.category_mlb = MultiLabelBinarizer()
        self.category_mlb.fit([category_list])
        self.used_api_mlb = MultiLabelBinarizer()
        self.used_api_mlb.fit([api_list])
        self.des_lens = []
        self.category_token = []
        for des in self.description:
            self.des_lens.append(len(des) if len(des) < 50 else 50)

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        description = self.description[index]
        category_tensor = torch.tensor(self.category_mlb.transform([self.category[index]]), dtype=torch.long).squeeze()
        used_api_tensor = torch.tensor(self.used_api_mlb.transform([self.used_api[index]]), dtype=torch.long).squeeze()
        des_len = torch.tensor(self.des_lens[index])
        category_token = torch.LongTensor(self.category_token[index])
        return torch.tensor(index).long(), torch.tensor(description).long(), category_tensor, used_api_tensor, des_len, category_token


class ApiDataset(Dataset):
    def __init__(self, all_api=False):
        super().__init__()
        with open(rootPath + '/data/api_name.json', 'r') as f:
            name = json.load(f)
        with open(rootPath + '/data/api_description.json', 'r') as f:
            description = json.load(f)
        with open(rootPath + '/data/api_category.json', 'r') as f:
            category = json.load(f)
        with open(rootPath + '/data/category_list.json', 'r') as f:
            category_list = json.load(f)
        with open(rootPath + '/data/mashup_name.json', 'r') as f:
            self.mashup = json.load(f)
        with open(rootPath + '/data/used_api_list.json', 'r') as f:
            used_api_list = json.load(f)
        if all_api:
            self.name = name
            self.description = description
            self.category = category
            self.used_api = []
            for api in self.name:
                self.used_api.append([api])
        else:
            self.name = used_api_list
            self.description = []
            self.category = []
            self.used_api = []
            for api in self.name:
                self.description.append(description[name.index(api)])
                self.category.append(category[name.index(api)])
                self.used_api.append([api])

        self.num_category = len(category_list)
        self.num_api = len(used_api_list)
        self.category_mlb = MultiLabelBinarizer()
        self.category_mlb.fit([category_list])
        self.used_api_mlb = MultiLabelBinarizer()
        self.used_api_mlb.fit([used_api_list])
        self.des_lens = []
        self.category_token = []
        for des in self.description:
            self.des_lens.append(len(des) if len(des) < 50 else 50)

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        description = self.description[index]
        category_tensor = torch.tensor(self.category_mlb.transform([self.category[index]]), dtype=torch.long).squeeze()
        used_api_tensor = torch.tensor(self.used_api_mlb.transform([self.name[index]]), dtype=torch.long).squeeze()
        des_len = torch.tensor(self.des_lens[index])
        category_token = torch.LongTensor(self.category_token[index])

        return torch.tensor(index).long(), \
               torch.tensor(description).long(), \
               category_tensor, \
               used_api_tensor, \
               des_len, category_token


class BPRDataset(Dataset):
    def __init__(self, sample_indices, neg_num):
        super(BPRDataset, self).__init__()
        self.ds = TextDataset()
        self.sample_indices = sample_indices
        self.triplet = None
        self.neg_num = neg_num  # 一个正例对应需要采样的负例数量
        self.create_triplet()

    def create_triplet(self):
        pairs = []
        triplet = []
        neg_list = list(range(len(self.ds.api_ds)))
        for sample in self.sample_indices:
            pos_indices = self.ds.mashup_ds[sample][3].nonzero().flatten().tolist()
            for pos in pos_indices:
                pairs.append([sample, pos])
        for pair in pairs:
            break_point = 0
            while(True):
                ch = choice(neg_list)
                if break_point == self.neg_num:
                    break
                elif ch != pair[1]:
                    triplet.append((pair[0], pair[1], ch))
                    break_point += 1

        self.triplet = triplet

    def __len__(self):
        return len(self.triplet)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample = self.triplet[index]
        mashup = self.ds.mashup_ds[sample[0]]
        api_i = self.ds.api_ds[sample[1]]
        api_j = self.ds.api_ds[sample[2]]
        return mashup, api_i, api_j


# class NNRDataset(Dataset):
#     def __init__(self, nn_num):
#         super(NNRDataset, self).__init__()
#         self.tds = TextDataset()
# 
#         # self.sample_indices = sample_indices
#         self.nn_num = nn_num  # 近邻mashup数量
#         self.sim_matrix = torch.zeros(self.nn_num, self.tds.embed_dim)
#         self.mashup_feature = torch.self.tds.embed[ds.mashup_ds.description].sum(dim=1)
#         self.sim_cal()
#
#     def sim_cal(self):
#         for i, des_list in enumerate(range(self.nn_num)):
#             tmp_ = torch.zeros(300)
#
#             m_feature[i] = torch.nn.functional.normalize(tmp_, dim=0)
#         self.sim_matrix = m_feature.mm(m_feature.t()).argsort(dim=1, descending=True)
#
#     def __len__(self):
#         return len(self.mds)
#
#     def __getitem__(self, index):
#         if torch.is_tensor(index):
#             index = index.tolist()
#         main_mashup = self.tds.mashup_ds[index]
#         nn_mashup_des = torch.zeros(self.nn_num, 50)
#         for count, i in enumerate(self.sim_matrix[index, 1:self.nn_num+1]):
#             nn_mashup_des[count] = self.tds.mashup_ds[i][1]
#
#         return main_mashup, nn_mashup_des.long()


class TextDataset:
    def __init__(self):
        cache = '.vec_cache'
        if not os.path.exists(cache):
            os.mkdir(cache)
        self.mashup_ds = MashupDataset()
        self.api_ds = ApiDataset()
        self.max_vocab_size = 10000
        self.max_doc_len = 50
        self.vectors = Vectors(name=rootPath + '/src/glove.6B.300d.txt', cache=cache)
        self.field = Field(sequential=True, tokenize=tokenize, lower=True, fix_length=self.max_doc_len)
        self.field.build_vocab(self.mashup_ds.description, self.api_ds.description, vectors=self.vectors, min_freq=1, max_size=self.max_vocab_size)
        self.random_seed = 2020
        self.num_category = self.mashup_ds.num_category
        self.num_mashup = len(self.mashup_ds)
        self.num_api = len(self.api_ds)
        self.vocab_size = len(self.field.vocab)
        self.embed = self.field.vocab.vectors
        self.embed_dim = self.vectors.dim
        self.des_lens = []
        self.word2id()
        self.tag2feature()

    def word2id(self):
        for i, des in enumerate(self.mashup_ds.description):
            tokens = [self.field.vocab.stoi[x] for x in des]
            if not tokens:
                tokens = [0]
            if len(tokens) < self.max_doc_len:
                tokens.extend([1] * (self.max_doc_len - len(tokens)))
            else:
                tokens = tokens[:self.max_doc_len]
            self.mashup_ds.description[i] = tokens

        for i, des in enumerate(self.api_ds.description):
            tokens = [self.field.vocab.stoi[x] for x in des]
            if not tokens:
                tokens = [0]
            if len(tokens) < self.max_doc_len:
                tokens.extend([1] * (self.max_doc_len - len(tokens)))
            else:
                tokens = tokens[:self.max_doc_len]
            self.api_ds.description[i] = tokens

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

        for i, category in enumerate(self.api_ds.category):
            tokens = [self.field.vocab.stoi[x] for x in tokenize(' '.join(category))]
            if not tokens:
                tokens = [0]
            if len(tokens) < 10:
                tokens.extend([1] * (10 - len(tokens)))
            else:
                tokens = tokens[:10]
            self.api_ds.category_token.append(tokens)


class F3RMDataset(Dataset):
    def __init__(self, nn_num=10):
        super(F3RMDataset, self).__init__()
        cache = '.vec_cache'
        if not os.path.exists(cache):
            os.mkdir(cache)
        self.tds = TextDataset()

        # self.sample_indices = sample_indices
        self.nn_num = nn_num  # 近邻mashup数量
        self.neighbor_mashup_des = torch.zeros(len(self.tds.mashup_ds), self.nn_num, self.tds.max_doc_len)
        self.mashup_feature = torch.nn.functional.normalize(self.tds.embed[self.tds.mashup_ds.description].sum(dim=1))
        self.sim = torch.nn.functional.normalize(torch.mm(self.mashup_feature, self.mashup_feature.t()))
        self.neighbor_mashup_index = self.sim.argsort(descending=True)[:, :self.nn_num]
        for i in range(len(self.tds.mashup_ds)):
            for j, index in enumerate(range(self.nn_num)):
                self.neighbor_mashup_des[i, j] = self.tds.mashup_ds[index][1]

    def __len__(self):
        return len(self.tds.mashup_ds)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        main_mashup = self.tds.mashup_ds[index]
        n_mashup_des = self.neighbor_mashup_des[index]
        return main_mashup, n_mashup_des.long()


class FCDataset(Dataset):
    def __init__(self, sample_indices, is_training=True):
        super(FCDataset, self).__init__()
        self.ds = TextDataset()
        self.triplet = []
        if is_training:
            self.neg_num = 14  # 一个正例对应需要采样的负例数量
            for indice in sample_indices:
                pos_indices = self.ds.mashup_ds[indice][3].nonzero().flatten().tolist()
                for pos in pos_indices:
                    self.triplet.append([indice, pos, 1])
                for idx in range(self.neg_num):
                    r = randint(0, 1646)
                    if r not in pos_indices:
                        self.triplet.append([indice, r, -1])
        else:
            for indice in sample_indices:
                pos_indices = self.ds.mashup_ds[indice][3].nonzero().flatten().tolist()
                for idx in range(len(self.ds.api_ds)):
                    if idx in pos_indices:
                        self.triplet.append([indice, idx, 1])
                    else:
                        self.triplet.append([indice, idx, -1])

    def __len__(self):
        return len(self.triplet)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample = self.triplet[index]
        mashup = self.ds.mashup_ds[sample[0]]
        api = self.ds.api_ds[sample[1]]
        label = sample[2]
        return mashup, api, label


if __name__ == '__main__':
    # mashup_ds = MashupDataset()
    # api_ds = ApiDataset()
    # ds = F3RMDataset()
    ds = TextDataset()
