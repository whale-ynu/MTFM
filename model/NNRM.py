# coding: UTF-8
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
from src.metrics import *
from src.utils import *


class Config(object):
    def __init__(self, ds_config):
        self.model_name = 'NNRM'
        self.embed_dim = ds_config.embed_dim
        self.max_doc_len = ds_config.max_doc_len
        self.dropout = 0.2
        self.num_category = ds_config.num_category
        self.hidden_size = 256
        self.num_layer = 1
        self.feature_size = 8
        self.num_kernel = 128
        self.dropout = 0.2
        self.kernel_size = [2, 3, 4, 5]
        self.num_mashup = ds_config.num_mashup
        self.num_api = ds_config.num_api
        self.vocab_size = ds_config.vocab_size
        self.embed = ds_config.embed
        self.lr = 1e-3
        self.batch_size = 128
        self.device = ('cuda:1' if torch.cuda.is_available() else 'cpu')


class NNRM(nn.Module):
    def __init__(self, config):
        super(NNRM, self).__init__()
        if config.embed is not None:
            self.m_embedding = nn.Embedding.from_pretrained(config.embed, freeze=False)
            self.a_embedding = nn.Embedding.from_pretrained(config.embed, freeze=False)
        else:
            self.m_embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.vocab_size - 1)
            self.a_embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.vocab_size - 1)
        self.m_convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=config.embed_dim,
                                    out_channels=config.num_kernel,
                                    kernel_size=h),
                          # nn.BatchNorm1d(num_features=config.feature_size),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=h))
            for h in config.kernel_size
        ])
        self.m_fc = nn.Linear(in_features=config.num_kernel * len(config.kernel_size),
                              out_features=config.num_api)
        # self.nn_conv = nn.Conv2d(in_channels=10, out_channels=config.num_kernel, kernel_size=(1, 3))
        self.nn_fc = nn.Linear(in_features=config.num_kernel * len(config.kernel_size) * 10, out_features=config.num_api)

        self.a_convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=config.embed_dim,
                                    out_channels=config.num_kernel,
                                    kernel_size=h),
                          # nn.BatchNorm1d(num_features=config.feature_size),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=h))
            for h in config.kernel_size
        ])
        self.a_fc = nn.Linear(in_features=config.num_kernel * len(config.kernel_size),
                              out_features=config.num_api)

        self.dropout = nn.Dropout(config.dropout)
        self.logistic = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, main_mashup_des, nn_mashup_des, api_des):
        m_embed = self.m_embedding(main_mashup_des)
        nn_embed = self.m_embedding(nn_mashup_des)
        a_embed = self.a_embedding(api_des)
        # batch_size x text_len x embedding_size  -> batch_size x embedding_size x text_len
        m_embed = m_embed.permute(0, 2, 1)
        a_embed = a_embed.permute(0, 2, 1)
        nn_embed = nn_embed.permute(0, 1, 3, 2)
        nn_input = nn_embed.view(-1, nn_embed.size(2), nn_embed.size(3))
        # print('embed size 2',embed_x.size())  # 32*256*35
        m_out = [conv(m_embed) for conv in self.m_convs]  # out[i]:batch_size x feature_size*1
        m_out = torch.cat(m_out, dim=2)  # 对应第二个维度（行）拼接起来，比如说5*2*1,5*3*1的拼接变成5*5*1
        m_out = m_out.view(-1, m_out.size(1))
        m_out = self.m_fc(m_out)

        nn_out = [conv(nn_input) for conv in self.m_convs]  # out[i]:batch_size x feature_size*1
        nn_out = torch.cat(nn_out, dim=2)  # 对应第二个维度（行）拼接起来，比如说5*2*1,5*3*1的拼接变成5*5*1
        nn_out = nn_out.view(nn_embed.size(0), nn_embed.size(1) * nn_out.size(1))
        nn_out = self.nn_fc(nn_out)

        a_out = [conv(a_embed) for conv in self.a_convs]  # out[i]:batch_size x feature_size*1
        a_out = torch.cat(a_out, dim=2)  # 对应第二个维度（行）拼接起来，比如说5*2*1,5*3*1的拼接变成5*5*1
        a_out = a_out.view(-1, a_out.size(1))
        a_out = self.a_fc(a_out)

        m_out = m_out + F.normalize(a_out) + F.normalize(nn_out)
        return self.logistic(m_out)


class Train(object):
    def __init__(self, model, config, train_iter, test_iter, val_iter, log, model_path=None):
        self.model = model
        self.config = config
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.val_iter = val_iter
        self.api_cri = torch.nn.BCELoss()
        self.cate_cri = torch.nn.BCELoss()
        self.optim = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        self.epoch = 10
        self.top_k = [5, 10, 15, 20, 25, 30, 35, 40]
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
            ap = np.zeros(len(self.top_k))
            pre = np.zeros(len(self.top_k))
            recall = np.zeros(len(self.top_k))
            ndcg = np.zeros(len(self.top_k))
            hd = np.zeros(len(self.top_k))
            api_loss = []
            num_batch = len(self.train_iter)
            for batch_idx, batch_data in enumerate(self.train_iter):
                # batch_data: index, des, category, used_api
                index = batch_data[0][0].to(self.config.device)
                main_mashup_des = batch_data[0][1].to(self.config.device)
                category_target = batch_data[0][2].float().to(self.config.device)
                api_target = batch_data[0][3].float().to(self.config.device)
                nn_mashup_des = batch_data[1].to(self.config.device)
                api_des = batch_data[2].to(self.config.device)
                self.optim.zero_grad()
                api_pred = self.model(main_mashup_des, nn_mashup_des, api_des)
                api_loss_ = self.api_cri(api_pred, api_target)
                api_loss_.backward()
                self.optim.step()
                api_loss.append(api_loss_.item())
            api_pred = api_pred.cpu().detach()
            for i, k in enumerate(self.top_k):
                ndcg[i] += get_ndcg(batch_data[0][3], api_pred, k=k)
                ap_, pre_, recall_ = get_map_pre_recall(batch_data[0][3], api_pred, k=k)
                ap[i] += ap_
                pre[i] += pre_
                recall[i] += recall_
                hd[i] += get_hd(api_pred, k=k)
            api_loss = np.average(api_loss)

            info = '[Epoch:%s] ApiLoss:%s \n' \
                   'NDCG:%s\n' \
                   'AP:%s\n' \
                   'Pre:%s\n' \
                   'Recall:%s\n' \
                   'HD:%s ' % (epoch+1, api_loss.round(6), ndcg.round(6), ap.round(6), pre.round(6), recall.round(6),
                                                           hd.round(6))
            print(info)
            self.log.write(info+'\n')
            self.log.flush()
            val_loss = self.evaluate()
            self.early_stopping(float(val_loss), self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break
        print('Testing')
        self.evaluate(test=True)

    def evaluate(self, test=False):
        if test:
            data_iter = self.test_iter
            label = 'Test'
        else:
            data_iter = self.val_iter
            label = 'Evaluate'
        self.model.eval()
        ap = np.zeros(len(self.top_k))
        pre = np.zeros(len(self.top_k))
        recall = np.zeros(len(self.top_k))
        ndcg = np.zeros(len(self.top_k))
        hd = np.zeros(len(self.top_k))
        api_loss = []
        num_batch = len(data_iter)
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_iter):
                index = batch_data[0][0].to(self.config.device)
                main_mashup_des = batch_data[0][1].to(self.config.device)
                category_target = batch_data[0][2].float().to(self.config.device)
                api_target = batch_data[0][3].float().to(self.config.device)
                nn_mashup_des = batch_data[1].to(self.config.device)
                api_des = batch_data[2].to(self.config.device)
                api_pred = self.model(main_mashup_des, nn_mashup_des, api_des)
                api_loss_ = self.api_cri(api_pred, api_target)
                api_loss.append(api_loss_.item())
                api_pred = api_pred.cpu().detach()
                for i, k in enumerate(self.top_k):
                    ndcg[i] += get_ndcg(batch_data[0][3], api_pred, k=k)
                    ap_, p_, r_ = get_map_pre_recall(batch_data[0][3], api_pred, k=k)
                    ap[i] += ap_
                    pre[i] += p_
                    recall[i] += r_
                    hd[i] += get_hd(api_pred, k=k)
        api_loss = np.average(api_loss)
        info = '[%s] ApiLoss:%s \n' \
               'NDCG:%s\n' \
               'AP:%s\n' \
               'Pre:%s\n' \
               'Recall:%s\n' \
               'HD:%s ' % (label, api_loss.round(6),(ndcg / num_batch).round(6), (ap / num_batch).round(6),
                                                       (pre / num_batch).round(6), (recall / num_batch).round(6),
                                                       (hd / num_batch).round(6))
        print(info)
        self.log.write(info+'\n')
        self.log.flush()
        return api_loss


if __name__ == '__main__':
    # load ds
    print('Start ...')
    start_time = time.time()
    now = time.time()
    ds = NNRDataset(nn_num=10)
    print('Time for loading dataset: ', get_time(now))

    # initial
    train_idx, val_idx, test_idx = get_indices(ds)
    config = Config(ds.tds)
    model = NNRM(config)
    model.to(config.device)
    train_iter = DataLoader(ds, batch_size=config.batch_size, sampler=SubsetRandomSampler(train_idx), drop_last=True)
    val_iter = DataLoader(ds, batch_size=config.batch_size, sampler=SubsetRandomSampler(val_idx), drop_last=True)
    test_iter = DataLoader(ds, batch_size=config.batch_size, sampler=SubsetRandomSampler(test_idx), drop_last=True)
    # training
    now = int(time.time())
    timeStruct = time.localtime(now)
    strTime = time.strftime("%Y-%m-%d-%H:%M:%S", timeStruct)
    log_path = 'log/{0}.log'.format(config.model_name)
    log = open(log_path, mode='a')
    log.write('{0} {1} \ndropout:{2}, lr:{3}, feature_size:{4}, batch_size:{5}\n'.format(config.model_name,
                                                                                         strTime,
                                                                                         str(config.dropout),
                                                                                         str(config.lr),
                                                                                         str(config.feature_size),
                                                                                         str(config.batch_size)))
    log.flush()
    # training
    train_func = Train(model=model,
                       config=config,
                       train_iter=train_iter,
                       test_iter=test_iter,
                       val_iter=val_iter,
                       log=log)
    train_func.train()
    # model_path = 'checkpoint/%s.pth' % config.model_name
    # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # case_path = 'case/{0}.json'.format(config.model_name)
    # case = open(case_path, mode='w')
    # mashup_case = []
    # with torch.no_grad():
    #     model.eval()
    #
    #     for batch_idx, batch_data in enumerate(test_iter):
    #         index = batch_data[0][0].to(config.device)
    #         main_mashup_des = batch_data[0][1].to(config.device)
    #         category_target = batch_data[0][2].float().to(config.device)
    #         api_target = batch_data[0][3].argsort(descending=True)[:, :3].tolist()
    #         nn_mashup_des = batch_data[1].to(config.device)
    #         api_des = batch_data[2].to(config.device)
    #         api_pred = model(main_mashup_des, nn_mashup_des, api_des)
    #         api_pred = api_pred.cpu().argsort(descending=True)[:, :3].tolist()
    #         for i, api_tuple in enumerate(zip(api_target, api_pred)):
    #             target = []
    #             pred = []
    #             name = ds.mds.name[index[i].cpu().tolist()]
    #             for t in api_tuple[0]:
    #                 target.append(ds.mds.used_api_mlb.classes_[t])
    #             for t in api_tuple[1]:
    #                 pred.append(ds.mds.used_api_mlb.classes_[t])
    #             mashup_case.append((name, target, pred))
    # json.dump(mashup_case, case)
    # case.close()
    log.close()