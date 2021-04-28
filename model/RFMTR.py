# -*- conding: utf-8 -*-
"""
@File   : RFMTR.py
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
import json
curPath = os.path.abspath(os.path.dirname('__file__'))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import time
from src.dataset_class import F3RMDataset
from src.metrics import *
from src.utils import *
from metric import metric


class Config:
    def __init__(self, ds_config):
        self.model_name = 'RFMTR'
        self.embed_dim = ds_config.embed_dim
        self.max_doc_len = ds_config.max_doc_len
        self.dropout = 0.2
        self.num_category = ds_config.num_category
        self.feature_size = 8
        self.num_kernel = 256
        self.kernel_size = [2, 3, 4, 5]
        self.num_mashup = ds_config.num_mashup
        self.num_api = ds_config.num_api
        self.vocab_size = ds_config.vocab_size
        self.embed = ds_config.embed
        self.lr = 1e-4
        self.batch_size = 128
        self.device = ('cuda:2' if torch.cuda.is_available() else 'cpu')


class RFMTR(nn.Module):
    def __init__(self, config):
        super(RFMTR, self).__init__()

        if config.embed is not None:
            self.m_embedding = nn.Embedding.from_pretrained(config.embed, freeze=False)
        else:
            self.m_embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.vocab_size - 1)
        self.feature_interaction = nn.Parameter(torch.FloatTensor(config.feature_size, config.num_api))

        self.m_convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=config.embed_dim,
                                    out_channels=config.num_kernel,
                                    kernel_size=h),
                          # nn.BatchNorm1d(num_features=config.feature_size),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=config.max_doc_len - h + 1))
            for h in config.kernel_size
        ])

        self.m_fc = nn.Linear(in_features=config.num_kernel * len(config.kernel_size), out_features=config.num_api)
        self.nn_fc = nn.Linear(in_features=config.num_kernel * len(config.kernel_size), out_features=config.num_api)
        self.feature_fc = nn.Linear(in_features=config.num_api, out_features=config.feature_size)
        self.category_fc = nn.Linear(in_features=config.num_api, out_features=config.num_category)

        self.mlp_1 = nn.Linear(in_features=(config.num_api + 1) * config.feature_size, out_features=config.num_api)
        self.mlp_2 = nn.Linear(in_features=config.num_api, out_features=config.num_api)
        self.mlp_3 = nn.Linear(in_features=config.num_api, out_features=config.num_category)

        self.dp_1 = nn.Linear(in_features=config.num_api, out_features=config.num_api)
        self.dp_2 = nn.Linear(in_features=config.num_api, out_features=config.num_category)

        self.cat_1 = nn.Linear(in_features=config.num_api * 3, out_features=config.num_api)
        self.cat_2 = nn.Linear(in_features=config.num_category * 3, out_features=config.num_category)

        self.dropout = nn.Dropout(config.dropout)
        self.logistic = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, main_mashup_des, nn_mashup_des):
        # embedding layer
        m_embed = self.m_embedding(main_mashup_des)
        nn_embed = self.m_embedding(nn_mashup_des)
        # batch_size x text_len x embedding_size  -> batch_size x embedding_size x text_len
        m_embed = m_embed.permute(0, 2, 1)
        nn_embed = nn_embed.permute(0, 1, 3, 2)
        nn_input = nn_embed.view(-1, nn_embed.size(2), nn_embed.size(3))

        # semantic encoder
        # target mashup description features
        m_out = [conv(m_embed) for conv in self.m_convs]  # out[i]:batch_size x feature_size*1
        m_out = torch.cat(m_out, dim=1)  # 对应第二个维度（行）拼接起来，比如说5*2*1,5*3*1的拼接变成5*5*1
        m_out = m_out.squeeze()
        m_out = self.m_fc(m_out)
        # neighbor mashups description features
        nn_out = [conv(nn_input) for conv in self.m_convs]  # out[i]:batch_size x feature_size*1
        nn_out = torch.cat(nn_out, dim=1)  # 对应第二个维度（行）拼接起来，比如说5*2*1,5*3*1的拼接变成5*5*1
        nn_out = nn_out.view(nn_embed.size(0), nn_embed.size(1), nn_out.size(1), nn_out.size(2))
        nn_out = nn_out.squeeze()
        nn_out = nn_out.sum(dim=1)
        nn_out = self.dropout(self.nn_fc(nn_out))

        # feature reinforcement module
        reinforced_feature = m_out + self.tanh(nn_out)
        reinforced_category = self.category_fc(reinforced_feature)

        # feature interaction
        # dot produce interaction
        inter_feature = self.feature_fc(reinforced_feature)
        dp = torch.mm(inter_feature, self.feature_interaction)
        dp_api = self.dp_1(dp)
        dp_category = self.dp_2(dp)

        # multi layer perception
        cat = torch.cat((inter_feature, self.feature_interaction.flatten().unsqueeze(dim=0).repeat(inter_feature.size(0), 1)), dim=1)
        mlp = self.mlp_1(cat)
        mlp_api = self.mlp_2(mlp)
        mlp_category = self.mlp_3(mlp)

        # feature fusion
        fusion_api = self.cat_1(torch.cat((reinforced_feature, dp_api, mlp_api), dim=1))
        fusion_category = self.cat_2(torch.cat((reinforced_category, dp_category, mlp_category), dim=1))

        return self.logistic(fusion_api), self.logistic(fusion_category)


class Train(object):
    def __init__(self, model, ds, config, train_iter, test_iter, val_iter, log, model_path=None):
        self.model = model
        self.ds = ds
        self.config = config
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.val_iter = val_iter
        self.api_cri = torch.nn.BCELoss()
        self.category_cri = torch.nn.BCELoss()
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
        for epoch in range(self.epoch):
            self.model.train()

            api_loss = []
            category_loss = []

            for batch_idx, batch_data in enumerate(self.train_iter):
                # batch_data: ((index, des, category, used_api, des_len), nn_mashup_des)
                index = batch_data[0][0].to(self.config.device)
                main_mashup_des = batch_data[0][1].to(self.config.device)
                category_target = batch_data[0][2].float().to(self.config.device)
                api_target = batch_data[0][3].float().to(self.config.device)
                nn_mashup_des = batch_data[1].to(self.config.device)
                self.optim.zero_grad()
                api_pred, category_pred = self.model(main_mashup_des, nn_mashup_des)
                api_loss_ = self.api_cri(api_pred, api_target)
                category_loss_ = self.category_cri(category_pred, category_target)
                loss_ = api_loss_ + category_loss_
                loss_.backward()
                self.optim.step()
                api_loss.append(api_loss_.item())
                category_loss.append(category_loss_.item())

            api_loss = np.average(api_loss)
            category_loss = np.average(category_loss)

            info = '[Epoch:%s] ApiLoss:%s CateLoss:%s' % (epoch + 1, api_loss.round(6), category_loss.round(6))
            print(info)
            self.log.write(info)
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
        # category
        ndcg_c = np.zeros(len(self.top_k_list))
        recall_c = np.zeros(len(self.top_k_list))
        ap_c = np.zeros(len(self.top_k_list))
        pre_c = np.zeros(len(self.top_k_list))

        api_loss = []
        category_loss = []
        num_batch = len(data_iter)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_iter):
                # batch_data: ((index, des, category, used_api, des_len), nn_mashup_des)
                index = batch_data[0][0].to(self.config.device)
                main_mashup_des = batch_data[0][1].to(self.config.device)
                category_target = batch_data[0][2].float().to(self.config.device)
                api_target = batch_data[0][3].float().to(self.config.device)
                nn_mashup_des = batch_data[1].to(self.config.device)
                api_pred, category_pred = self.model(main_mashup_des, nn_mashup_des)
                api_loss_ = self.api_cri(api_pred, api_target)
                category_loss_ = self.category_cri(category_pred, category_target)
                api_loss.append(api_loss_.item())
                category_loss.append(category_loss_.item())

                api_pred = api_pred.cpu().detach()
                category_pred = category_pred.cpu().detach()

                ndcg_, recall_, ap_, pre_ = metric(batch_data[3], api_pred.cpu(),
                                                   top_k_list=self.top_k_list)
                ndcg_a += ndcg_
                recall_a += recall_
                ap_a += ap_
                pre_a += pre_

                ndcg_, recall_, ap_, pre_ = metric(batch_data[2], category_pred.cpu(),
                                                   top_k_list=self.top_k_list)
                ndcg_c += ndcg_
                recall_c += recall_
                ap_c += ap_
                pre_c += pre_

        api_loss = np.average(api_loss)
        category_loss = np.average(category_loss)

        ndcg_a /= num_batch
        recall_a /= num_batch
        ap_a /= num_batch
        pre_a /= num_batch
        ndcg_c /= num_batch
        recall_c /= num_batch
        ap_c /= num_batch
        pre_c /= num_batch

        info = '[%s] ApiLoss:%s CateLoss:%s\n' \
               'NDCG_A:%s\n' \
               'AP_A:%s\n' \
               'Pre_A:%s\n' \
               'Recall_A:%s\n' \
               'NDCG_C:%s\n' \
               'AP_C:%s\n' \
               'Pre_C:%s\n' \
               'Recall_C:%s' % (
                   label, api_loss.round(6), category_loss.round(6), ndcg_a.round(6), ap_a.round(6), pre_a.round(6),
                   recall_a.round(6), ndcg_c.round(6), ap_c.round(6), pre_c.round(6), recall_c.round(6))
        print(info)
        self.log.write(info+'\n')
        self.log.flush()
        return api_loss

    def case_analysis(self):
        case_path = 'case/{0}.json'.format(self.config.model_name)
        a_case = open(case_path, mode='w')
        case_path = 'case/{0}_c.json'.format(self.config.model_name)
        c_case = open(case_path, mode='w')
        api_case = []
        cate_case = []
        self.model.eval()
        with torch.no_grad():

            for batch_idx, batch_data in enumerate(self.test_iter):
                # batch_data: ((index, des, category, used_api, des_len), nn_mashup_des)
                index = batch_data[0][0].to(self.config.device)
                main_mashup_des = batch_data[0][1].to(self.config.device)
                category_target = batch_data[0][2].argsort(descending=True)[:, :3].tolist()
                api_target = batch_data[0][3].argsort(descending=True)[:, :3].tolist()
                nn_mashup_des = batch_data[1].to(self.config.device)
                api_pred_, category_pred_ = self.model(main_mashup_des, nn_mashup_des)
                api_pred_ = api_pred_.cpu().argsort(descending=True)[:, :3].tolist()
                category_pred_ = category_pred_.cpu().argsort(descending=True)[:, :3].tolist()
                for i, api_tuple in enumerate(zip(api_target, api_pred_)):
                    target = []
                    pred = []
                    name = self.ds.mashup_ds.name[index[i].cpu().tolist()]
                    for t in api_tuple[0]:
                        target.append(self.ds.mashup_ds.used_api_mlb.classes_[t])
                    for t in api_tuple[1]:
                        pred.append(self.ds.mashup_ds.used_api_mlb.classes_[t])
                    api_case.append((name, target, pred))
                for i, cate_tuple in enumerate(zip(category_target, category_pred_)):
                    target = []
                    pred = []
                    name = self.ds.mashup_ds.name[index[i].cpu().tolist()]
                    for t in cate_tuple[0]:
                        target.append(self.ds.mashup_ds.category_mlb.classes_[t])
                    for t in cate_tuple[1]:
                        pred.append(self.ds.mashup_ds.category_mlb.classes_[t])
                    cate_case.append((name, target, pred))
        json.dump(api_case, a_case)
        json.dump(cate_case, c_case)
        a_case.close()
        c_case.close()


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
    model = RFMTR(config)
    model.to(config.device)
    train_iter = DataLoader(ds, batch_size=config.batch_size, sampler=SubsetRandomSampler(train_idx), drop_last=True)
    val_iter = DataLoader(ds, batch_size=config.batch_size, sampler=SubsetRandomSampler(val_idx), drop_last=True)
    test_iter = DataLoader(ds, batch_size=1, sampler=SubsetRandomSampler(test_idx), drop_last=True)
    # training
    now = int(time.time())
    log_path = 'log/{0}.log'.format(config.model_name)
    log = open(log_path, mode='a')
    log.write(strftime)
    log.flush()
    # training
    train_func = Train(model=model,
                       ds=ds.tds,
                       config=config,
                       train_iter=train_iter,
                       test_iter=test_iter,
                       val_iter=val_iter,
                       log=log)
    train_func.train()
    train_func.evaluate(test=True)
    # testing
    # model_path = 'checkpoint/%s.pth' % config.model_name
    # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    train_func.case_analysis()

    log.close()
