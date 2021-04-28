# -*- conding: utf-8 -*-
"""
@File   : MTM.py
@Time   : 2021/3/15
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
from src.dataset_class import *
from src.utils import *
from metric import metric, metric2


class MTMConfig(object):
    def __init__(self, ds_config):
        self.model_name = 'MTM'
        self.embed_dim = ds_config.embed_dim
        self.max_doc_len = ds_config.max_doc_len
        self.dropout = 0.2
        self.num_category = ds_config.num_category
        self.feature_dim = 8
        self.num_kernel = 256
        self.dropout = 0.2
        self.kernel_size = [2, 3, 4, 5]
        self.num_mashup = ds_config.num_mashup
        self.num_api = ds_config.num_api
        self.vocab_size = ds_config.vocab_size
        self.embed = ds_config.embed
        self.lr = 1e-4
        self.batch_size = 128
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')


class MTM(nn.Module):
    def __init__(self, config):
        super(MTM, self).__init__()
        if config.embed is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embed, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.vocab_size - 1)

        self.sc_convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=config.embed_dim,
                                    out_channels=config.num_kernel,
                                    kernel_size=h),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=config.max_doc_len-h+1))
            for h in config.kernel_size
        ])
        self.sc_fcl = nn.Linear(in_features=config.num_kernel * len(config.kernel_size),
                                out_features=config.num_api)

        self.api_task_layer = nn.Linear(config.num_api, config.num_api)
        self.category_task_layer = nn.Linear(config.num_api, config.num_category)

        self.dropout = nn.Dropout(config.dropout)
        self.logistic = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, mashup_des):
        # semantic component
        embed = self.embedding(mashup_des)
        embed = embed.permute(0, 2, 1)
        e = [conv(embed) for conv in self.sc_convs]
        e = torch.cat(e, dim=2)
        e = e.view(e.size(0), -1)
        u_sc = self.sc_fcl(e)
        u_sc = self.tanh(u_sc)
        u_sc = self.dropout(u_sc)

        # api-specific task layer
        y_m = self.api_task_layer(u_sc)

        # mashup category-specific task layer
        z_m = self.category_task_layer(u_sc)

        return self.logistic(y_m), self.logistic(z_m)


class Train(object):
    def __init__(self, input_model, input_config, train_iter, test_iter, val_iter, log, input_ds, model_path=None):
        self.model = input_model
        self.config = input_config
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.val_iter = val_iter
        self.api_cri = torch.nn.BCELoss()
        self.cate_cri = torch.nn.BCELoss()
        self.optim = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        self.epoch = 100
        self.top_k_list = [1, 5, 10, 15, 20, 25, 30]
        self.log = log
        self.ds = input_ds
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = 'checkpoint/%s.pth' % self.config.model_name
        self.early_stopping = EarlyStopping(patience=7, path=self.model_path)

    def train(self):

        data_iter = self.train_iter
        self.model.train()
        print('Start training ...')

        for epoch in range(self.epoch):

            api_loss = []
            category_loss = []

            for batch_idx, batch_data in enumerate(data_iter):
                # batch_data: index, des, category_performance, used_api
                index = batch_data[0].to(self.config.device)
                des = batch_data[1].to(self.config.device)
                category_target = batch_data[2].float().to(self.config.device)
                api_target = batch_data[3].float().to(self.config.device)

                self.optim.zero_grad()
                api_pred, category_pred = self.model(des)

                api_loss_ = self.api_cri(api_pred, api_target)
                category_loss_ = self.cate_cri(category_pred, category_target)
                loss_ = category_loss_ + api_loss_
                loss_.backward()
                self.optim.step()
                api_loss.append(api_loss_.item())
                category_loss.append(category_loss_.item())

            api_loss = np.average(api_loss)
            category_loss = np.average(category_loss)

            info = '[Epoch:%s] ApiLoss:%s CateLoss:%s' % (epoch+1, api_loss.round(6), category_loss.round(6))
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
        # category_performance
        ndcg_c = np.zeros(len(self.top_k_list))
        recall_c = np.zeros(len(self.top_k_list))
        ap_c = np.zeros(len(self.top_k_list))
        pre_c = np.zeros(len(self.top_k_list))

        api_loss = []
        category_loss = []

        num_batch = len(data_iter)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_iter):
                index = batch_data[0].to(self.config.device)
                des = batch_data[1].to(self.config.device)
                category_target = batch_data[2].float().to(self.config.device)
                api_target = batch_data[3].float().to(self.config.device)
                api_pred, category_pred = self.model(des)
                api_loss_ = self.api_cri(api_pred, api_target)
                category_loss_ = self.cate_cri(category_pred, category_target)
                api_loss.append(api_loss_.item())
                category_loss.append(category_loss_.item())

                api_pred = api_pred.cpu().detach()
                category_pred = category_pred.cpu().detach()

                ndcg_, recall_, ap_, pre_ = metric(batch_data[3], api_pred.cpu(), top_k_list=self.top_k_list)
                ndcg_a += ndcg_
                recall_a += recall_
                ap_a += ap_
                pre_a += pre_

                ndcg_, recall_, ap_, pre_ = metric(batch_data[2], category_pred.cpu(), top_k_list=self.top_k_list)
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
        return api_loss + category_loss

    def case_analysis(self):
        case_path = 'case/{0}.json'.format(config.model_name)
        a_case = open(case_path, mode='w')
        case_path = 'case/{0}_c.json'.format(config.model_name)
        c_case = open(case_path, mode='w')
        api_case = []
        cate_case = []
        self.model.eval()
        with torch.no_grad():

            for batch_idx, batch_data in enumerate(self.test_iter):
                index = batch_data[0].to(self.config.device)
                des = batch_data[1].to(self.config.device)
                category_target = batch_data[2].argsort(descending=True)[:, :3].tolist()
                api_target = batch_data[3].argsort(descending=True)[:, :3].tolist()
                api_pred_, category_pred_ = self.model(des)
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
    ds = TextDataset()
    print('Time for loading dataset: ', get_time(now))

    # initial
    train_idx, val_idx, test_idx = get_indices(ds.mashup_ds)
    config = MTMConfig(ds)
    model = MTM(config)
    model.to(config.device)
    train_iter = DataLoader(ds.mashup_ds, batch_size=config.batch_size,
                            sampler=SubsetRandomSampler(train_idx), drop_last=True)
    val_iter = DataLoader(ds.mashup_ds, batch_size=config.batch_size,
                          sampler=SubsetRandomSampler(val_idx), drop_last=True)
    test_iter = DataLoader(ds.mashup_ds, batch_size=1,
                           sampler=SubsetRandomSampler(test_idx), drop_last=True)

    # training
    now = int(time.time())
    timeStruct = time.localtime(now)
    strTime = time.strftime("%Y-%m-%d", timeStruct)
    log_path = 'log/{0}.log'.format(config.model_name)
    log = open(log_path, mode='a')
    log.write(strTime + '\n')
    log.flush()

    # model_path = 'checkpoint/%s.pth' % config.model_name
    # model.load_state_dict(torch.load(model_path, map_location=config.device))
    train_func = Train(input_model=model,
                       input_config=config,
                       train_iter=train_iter,
                       test_iter=test_iter,
                       val_iter=val_iter,
                       log=log,
                       input_ds=ds)
    # training
    train_func.train()

    # testing
    train_func.evaluate(test=True)

    train_func.case_analysis()
    log.close()
