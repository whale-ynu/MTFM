# -*- conding: utf-8 -*-
"""
@File   : F3RM.py
@Time   : 2021/1/10
@Author : yhduan
@Desc   : None
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import os
import sys
curPath = os.path.abspath(os.path.dirname('__file__'))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import time
from src.dataset_class import *
from src.utils import *
from metric import metric


class Config:
    def __init__(self, ds_config):
        self.model_name = 'F3RM'
        self.embed_dim = ds_config.embed_dim
        self.max_doc_len = ds_config.max_doc_len
        self.dropout = 0.2
        self.num_category = ds_config.num_category
        self.feature_size = 4
        self.num_kernel = 64
        self.kernel_size = [2, 3, 4, 5]
        self.num_mashup = ds_config.num_mashup
        self.num_api = ds_config.num_api
        self.vocab_size = ds_config.vocab_size
        self.embed = ds_config.embed
        self.lr = 1e-3
        self.batch_size = 128
        self.device = ('cuda:1' if torch.cuda.is_available() else 'cpu')


class F3RM(nn.Module):
    def __init__(self, config):
        super(F3RM, self).__init__()
        if config.embed is not None:
            self.m_embedding = nn.Embedding.from_pretrained(config.embed, freeze=False)
        else:
            self.m_embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.vocab_size - 1)
        self.c_embedding = nn.Parameter(torch.FloatTensor(config.num_category, config.feature_size))
        self.m_convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=config.embed_dim,
                                    out_channels=config.num_kernel,
                                    kernel_size=h),
                          # nn.BatchNorm1d(num_features=config.feature_size),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=config.max_doc_len - h + 1))
            for h in config.kernel_size
        ])
        self.m_fc = nn.Linear(in_features=config.num_kernel * len(config.kernel_size),
                              out_features=config.num_api)
        # self.nn_conv = nn.Conv2d(in_channels=10, out_channels=config.num_kernel, kernel_size=(1, 3))
        self.nn_fc = nn.Linear(in_features=config.num_kernel * len(config.kernel_size), out_features=config.num_api)

        self.c_fc = nn.Linear(in_features=config.feature_size, out_features=config.num_api)
        self.fc_out = nn.Linear(in_features=config.num_api, out_features=config.num_api)
        self.dropout = nn.Dropout(config.dropout)
        self.logistic = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, main_mashup_des, nn_mashup_des, category):
        m_embed = self.m_embedding(main_mashup_des)
        nn_embed = self.m_embedding(nn_mashup_des)
        c_embed = torch.mm(category, self.c_embedding)
        # batch_size x text_len x embedding_size  -> batch_size x embedding_size x text_len
        m_embed = m_embed.permute(0, 2, 1)
        nn_embed = nn_embed.permute(0, 1, 3, 2)
        nn_input = nn_embed.view(-1, nn_embed.size(2), nn_embed.size(3))

        # print('embed size 2',embed_x.size())  # 32*256*35
        m_out = [conv(m_embed) for conv in self.m_convs]  # out[i]:batch_size x feature_size*1
        m_out = torch.cat(m_out, dim=1)  # 对应第二个维度（行）拼接起来，比如说5*2*1,5*3*1的拼接变成5*5*1
        m_out = m_out.squeeze()
        m_out = self.m_fc(m_out)

        nn_out = [conv(nn_input) for conv in self.m_convs]  # out[i]:batch_size x feature_size*1
        nn_out = torch.cat(nn_out, dim=1)  # 对应第二个维度（行）拼接起来，比如说5*2*1,5*3*1的拼接变成5*5*1

        nn_out = nn_out.view(nn_embed.size(0), nn_embed.size(1), nn_out.size(1), nn_out.size(2))
        nn_out = nn_out.squeeze()
        nn_out = nn_out.sum(dim=1)
        nn_out = self.dropout(self.nn_fc(nn_out))

        c_out = self.c_fc(c_embed)

        m_out = m_out + self.tanh(nn_out) + self.tanh(c_out)
        out = self.fc_out(m_out)
        return self.logistic(out)


class Train(object):
    def __init__(self, model, ds, config, train_iter, test_iter, val_iter, log, model_path=None):
        self.model = model
        self.ds = ds
        self.config = config
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.val_iter = val_iter
        self.api_cri = torch.nn.BCELoss()
        self.optim = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        self.epoch = 100
        self.top_k_list = [1, 5, 10, 15, 20, 25, 30]
        self.log = log
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = 'checkpoint/%s.pth' % self.config.model_name
        self.early_stopping = EarlyStopping(patience=7, path=self.model_path)

    def train(self):
        print('Start training ...')
        self.model.train()
        for epoch in range(self.epoch):
            # ap = np.zeros(len(self.top_k))
            # pre = np.zeros(len(self.top_k))
            # recall = np.zeros(len(self.top_k))
            # ndcg = np.zeros(len(self.top_k))
            # hd = np.zeros(len(self.top_k))
            api_loss = []
            num_batch = len(self.train_iter)
            for batch_idx, batch_data in enumerate(self.train_iter):
                # batch_data: ((index, des, category_performance, used_api, des_len), nn_mashup_des)
                index = batch_data[0][0].to(self.config.device)
                main_mashup_des = batch_data[0][1].to(self.config.device)
                category_input = batch_data[0][2].float().to(self.config.device)
                api_target = batch_data[0][3].float().to(self.config.device)
                nn_mashup_des = batch_data[1].to(self.config.device)
                self.optim.zero_grad()
                api_pred = self.model(main_mashup_des, nn_mashup_des, category_input)
                api_loss_ = self.api_cri(api_pred, api_target)
                api_loss_.backward()
                self.optim.step()
                api_loss.append(api_loss_.item())
            # for i, k in enumerate(self.top_k):
            #     ndcg[i] += get_ndcg(batch_data[0][3], api_pred, k=k)
            #     ap_, pre_, recall_ = get_map_pre_recall(batch_data[0][3], api_pred, k=k)
            #     ap[i] += ap_
            #     pre[i] += pre_
            #     recall[i] += recall_
            #     hd[i] += get_hd(api_pred, k=k)
            api_loss = np.average(api_loss)

            info = '[Epoch:%s] Loss:%s \n' % (epoch+1, api_loss.round(6))
            print(info)
            self.log.write(info+'\n')
            self.log.flush()
            val_loss = self.evaluate()
            self.early_stopping(float(val_loss), self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

    def evaluate(self, test=False):
        if test:
            data_iter = self.test_iter
            label = 'Test'
            print('Start testing ...')
        else:
            data_iter = self.val_iter
            label = 'Evaluate'
        self.model.eval()

        # API
        ndcg_a = np.zeros(len(self.top_k_list))
        recall_a = np.zeros(len(self.top_k_list))
        ap_a = np.zeros(len(self.top_k_list))
        pre_a = np.zeros(len(self.top_k_list))
        api_loss = []
        num_batch = len(data_iter)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_iter):
                # batch_data: ((index, des, category_performance, used_api, des_len), nn_mashup_des)
                index = batch_data[0][0].to(self.config.device)
                main_mashup_des = batch_data[0][1].to(self.config.device)
                category_input = batch_data[0][2].float().to(self.config.device)
                api_target = batch_data[0][3].float().to(self.config.device)
                nn_mashup_des = batch_data[1].to(self.config.device)
                api_pred = self.model(main_mashup_des, nn_mashup_des, category_input)
                api_loss_ = self.api_cri(api_pred, api_target)

                api_loss.append(api_loss_.item())

                ndcg_, recall_, ap_, pre_ = metric(batch_data[0][3], api_pred.cpu(),
                                                   top_k_list=self.top_k_list)
                ndcg_a += ndcg_
                recall_a += recall_
                ap_a += ap_
                pre_a += pre_
        api_loss = np.average(api_loss)
        ndcg_a /= num_batch
        recall_a /= num_batch
        ap_a /= num_batch
        pre_a /= num_batch
        info = '[%s] ApiLoss:%s\n' \
               'NDCG_A:%s\n' \
               'AP_A:%s\n' \
               'Pre_A:%s\n' \
               'Recall_A:%s\n' % (label, api_loss.round(6), ndcg_a.round(6),
                                  ap_a.round(6), pre_a.round(6), recall_a.round(6))
        print(info)
        self.log.write(info+'\n')
        self.log.flush()
        return api_loss

    def case_analysis(self):
        case_path = 'case/{0}.json'.format(self.config.model_name)
        case = open(case_path, mode='w')
        mashup_case = []
        with torch.no_grad():
            model.eval()

            for batch_idx, batch_data in enumerate(test_iter):
                index = batch_data[0][0].to(self.config.device)
                main_mashup_des = batch_data[0][1].to(self.config.device)
                category_input = batch_data[0][2].float().to(self.config.device)
                api_target = batch_data[0][3].argsort(descending=True)[:, :3].tolist()
                nn_mashup_des = batch_data[1].to(config.device)
                api_pred = model(main_mashup_des, nn_mashup_des, category_input)
                api_pred = api_pred.cpu().argsort(descending=True)[:, :3].tolist()

                for i, api_tuple in enumerate(zip(api_target, api_pred)):
                    target = []
                    pred = []
                    name = self.ds.mashup_ds.name[index[i].cpu().tolist()]
                    for t in api_tuple[0]:
                        target.append(self.ds.mashup_ds.used_api_mlb.classes_[t])
                    for t in api_tuple[1]:
                        pred.append(self.ds.mashup_ds.used_api_mlb.classes_[t])
                    mashup_case.append((name, target, pred))
        json.dump(mashup_case, case)
        case.close()


if __name__ == '__main__':
    # load ds
    print('Start ...')
    start_time = time.time()
    now = time.time()
    ds = F3RMDataset()
    print('Time for loading dataset: ', get_time(now))
    strftime = time.strftime("%Y-%m-%d", time.localtime())

    # initial
    train_idx, val_idx, test_idx = get_indices(ds)
    config = Config(ds.tds)
    model = F3RM(config)
    model.to(config.device)
    train_iter = DataLoader(ds, batch_size=config.batch_size, sampler=SubsetRandomSampler(train_idx), drop_last=True)
    val_iter = DataLoader(ds, batch_size=config.batch_size, sampler=SubsetRandomSampler(val_idx), drop_last=True)
    test_iter = DataLoader(ds, batch_size=1, sampler=SubsetRandomSampler(test_idx), drop_last=True)
    now = int(time.time())
    log_path = 'log/{0}.log'.format(config.model_name)
    log = open(log_path, mode='a')
    log.write(strftime)
    log.flush()

    # model_path = 'checkpoint/%s.pth' % config.model_name
    # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # training
    train_func = Train(model=model,
                       ds=ds.tds,
                       config=config,
                       train_iter=train_iter,
                       test_iter=test_iter,
                       val_iter=val_iter,
                       log=log)
    train_func.train()

    # testing
    train_func.evaluate(test=True)
    train_func.case_analysis()
    log.close()
