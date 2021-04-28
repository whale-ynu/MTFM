#%%
import os
import sys
import torch
import time
curPath = os.path.abspath(os.path.dirname('__file__'))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.dataset_class import TextDataset, FCDataset
from src.utils import EarlyStopping, get_indices
from metric import metric2
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Config:
    def __init__(self):
        self.model_name = 'FCLSTM'
        self.ds = TextDataset()
        self.max_doc_len = self.ds.max_doc_len
        self.max_vocab_size = self.ds.max_vocab_size
        self.embed_dim = self.ds.embed_dim
        self.num_layer = 2
        self.hidden_size = 128
        self.num_category = self.ds.num_category
        self.num_mashup = self.ds.num_mashup
        self.num_api = self.ds.num_api
        self.device = ('cuda:3' if torch.cuda.is_available() else 'cpu')
        self.dropout = 0.2
        self.batch_size = 128
        self.lr = 0.05


class FCLSTM(nn.Module):
    def __init__(self, input_config):
        super(FCLSTM, self).__init__()
        self.mashup_embed = nn.Embedding.from_pretrained(input_config.ds.embed, freeze=False)
        self.service_embed = nn.Embedding.from_pretrained(input_config.ds.embed, freeze=False)

        self.mashup_lstm = nn.LSTM(input_size=input_config.embed_dim, hidden_size=input_config.hidden_size,
                                   num_layers=input_config.num_layer, bidirectional=True,
                                   batch_first=True)
        self.service_lstm = nn.LSTM(input_size=input_config.embed_dim, hidden_size=input_config.hidden_size,
                                    num_layers=input_config.num_layer, bidirectional=True,
                                    batch_first=True)

        self.mashup_mlp = nn.Sequential(
            nn.Linear(input_config.embed_dim, input_config.hidden_size*2),
            nn.Sigmoid(),
        )
        self.service_mlp = nn.Sequential(
            nn.Linear(input_config.embed_dim, input_config.hidden_size*2),
            nn.Sigmoid(),
        )

        # self.mashup_fc = nn.Linear(input_config.hidden_size*2, input_config.feature_dim)
        # self.service_fc = nn.Linear(input_config.hidden_size*2, input_config.feature_dim)

        self.tanh = nn.Tanh()
        self.mashup_w = nn.Parameter(torch.zeros(input_config.hidden_size * 2))
        self.service_w = nn.Parameter(torch.zeros(input_config.hidden_size * 2))
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(input_config.num_api*2, input_config.num_api)


    def forward(self, mashup_des, mashup_des_len, mashup_tag, service_des, service_des_len, service_tag):
        mashup_embed = self.mashup_embed(mashup_des)
        packed = pack_padded_sequence(mashup_embed, mashup_des_len, batch_first=True, enforce_sorted=False)
        H, _ = self.mashup_lstm(packed)
        H, _ = pad_packed_sequence(H, batch_first=True)
        M = self.tanh(H)
        alpha = F.softmax(torch.matmul(M, self.mashup_w), dim=1).unsqueeze(-1)
        out = H * alpha
        out = torch.sum(out, dim=1)

        mashup_tag_embed = self.mashup_embed(mashup_tag)
        mashup_tag_mlp = self.mashup_mlp(mashup_tag_embed)
        mashup_tag_mlp = mashup_tag_mlp.sum(dim=1).squeeze()

        mashup_att = torch.mul(out, mashup_tag_mlp)
        mashup_feature = torch.cat((mashup_att, mashup_tag_mlp), dim=1)

        service_embed = self.service_embed(service_des)
        packed = pack_padded_sequence(service_embed, service_des_len, batch_first=True, enforce_sorted=False)
        H, _ = self.mashup_lstm(packed)
        H, _ = pad_packed_sequence(H, batch_first=True)
        M = self.tanh(H)
        alpha = F.softmax(torch.matmul(M, self.service_w), dim=1).unsqueeze(-1)
        out = H * alpha
        out = torch.sum(out, dim=1)

        service_tag_embed = self.service_embed(service_tag)
        service_tag_mlp = self.service_mlp(service_tag_embed)
        service_tag_mlp = service_tag_mlp.sum(dim=1).squeeze()

        service_att = torch.mul(out, service_tag_mlp)
        service_feature = torch.cat((service_att, service_tag_mlp), dim=1)

        y = F.cosine_similarity(mashup_feature, service_feature)
        return self.sigmoid(y)


class Train:
    def __init__(self, input_model, input_config, train_iter, test_iter, val_iter, input_log, model_path=None):
        self.config = input_config
        self.model = input_model.to(self.config.device)
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.val_iter = val_iter
        self.optim = torch.optim.SGD(self.model.parameters(), lr=self.config.lr)
        self.criterion = nn.HingeEmbeddingLoss()
        self.epoch = 100
        self.top_k_list = [1, 5, 10, 15, 20, 25, 30]
        self.log = input_log
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = 'checkpoint/%s.pth' % self.config.model_name
        self.early_stopping = EarlyStopping(patience=7, path=self.model_path)


    def train(self):

        print('Start training ...')
        for epoch in range(self.epoch):
            loss = []
            self.model.train()
            for batch_idx, batch_data in enumerate(self.train_iter):
                mashup_des = batch_data[0][1].to(self.config.device)
                des_len = batch_data[0][4].to(self.config.device)
                category_token = batch_data[0][5].to(self.config.device)
                api_des = batch_data[1][1].to(self.config.device)
                api_des_len = batch_data[1][4].to(self.config.device)
                api_category_token = batch_data[1][5].to(self.config.device)
                target = batch_data[2].float().to(self.config.device)

                self.optim.zero_grad()
                pred = self.model(mashup_des, des_len, category_token,
                                      api_des, api_des_len, api_category_token)
                loss_ = self.criterion(pred, target)
                loss_.backward()
                self.optim.step()
                loss.append(loss_.item())

            loss = np.average(loss)

            info = '[Epoch:%s] Loss:%s' % (epoch+1, loss.round(6))
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

        ap = np.zeros(len(self.top_k_list))
        pre = np.zeros(len(self.top_k_list))
        recall = np.zeros(len(self.top_k_list))
        ndcg = np.zeros(len(self.top_k_list))
        loss = []
        num_batch = len(data_iter)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_iter):
                mashup_des = batch_data[0][1].to(self.config.device)
                des_len = batch_data[0][4].to(self.config.device)
                category_token = batch_data[0][5].to(self.config.device)
                api_des = batch_data[1][1].to(self.config.device)
                api_des_len = batch_data[1][4].to(self.config.device)
                api_category_token = batch_data[1][5].to(self.config.device)
                target = batch_data[2].float().to(self.config.device)

                pred = self.model(mashup_des, des_len, category_token,
                                  api_des, api_des_len, api_category_token)
                loss_ = self.criterion(pred, target)
                loss.append(loss_.item())

                ndcg_, recall_, ap_, pre_ = metric2(batch_data[2].gt(0).long(), pred.cpu(), top_k_list=self.top_k_list)
                ndcg += ndcg_
                recall += recall_
                ap += ap_
                pre += pre_

            loss = np.average(loss)

        ndcg /= num_batch
        recall /= num_batch
        ap /= num_batch
        pre /= num_batch
        info = '[%s] ApiLoss:%s \n' \
               'NDCG:%s\n' \
               'AP:%s\n' \
               'Pre:%s\n' \
               'Recall:%s\n' % (label, loss.round(6), ndcg.round(6), ap.round(6), pre.round(6), recall.round(6))
        print(info)
        if label == 'Test':
            self.log.write(info + '\n')
            self.log.flush()
        return loss

    def case_analysis(self):
        case_path = 'case/{0}.json'.format(self.config.model_name)
        a_case = open(case_path, mode='w')
        case = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.test_iter):
                index = batch_data[0][0]
                mashup_des = batch_data[0][1].to(self.config.device)
                des_len = batch_data[0][4].to(self.config.device)
                category_token = batch_data[0][5].to(self.config.device)
                api_des = batch_data[1][1].to(self.config.device)
                api_des_len = batch_data[1][4].to(self.config.device)
                api_category_token = batch_data[1][5].to(self.config.device)
                target = batch_data[2].gt(0).long()

                pred = self.model(mashup_des, des_len, category_token,
                                  api_des, api_des_len, api_category_token)

                pred = pred.cpu().argsort(descending=True)[:, :3].tolist()

                for i, api_tuple in enumerate(zip(target, pred)):
                    target = []
                    pred = []
                    name = self.config.ds.mashup_ds.name[index[i].cpu().tolist()]
                    for t in api_tuple[0]:
                        target.append(self.config.ds.mashup_ds.used_api_mlb.classes_[t])
                    for t in api_tuple[1]:
                        pred.append(self.config.ds.mashup_ds.used_api_mlb.classes_[t])
                    case.append((name, target, pred))
        json.dump(case, a_case)
        a_case.close()


if __name__ == '__main__':
    # load ds
    print('Start ...')
    config = Config()
    model = FCLSTM(input_config=config)
    log_path = './log/%s.log' % config.model_name
    log = open(log_path, mode='a')
    now = int(time.time())
    timeStruct = time.localtime(now)
    strTime = time.strftime("%Y-%m-%d\n", timeStruct)
    train_idx, val_idx, test_idx = get_indices(config.ds.mashup_ds)
    idx = [train_idx, val_idx, test_idx]
    train_ds = FCDataset(train_idx)
    test_ds = FCDataset(test_idx, is_training=False)
    val_ds = FCDataset(val_idx, is_training=False)

    train_iter = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4)
    test_iter = DataLoader(test_ds, batch_size=len(test_idx), num_workers=4)
    val_iter = DataLoader(val_ds, batch_size=len(val_idx), num_workers=4)

    train_func = Train(input_model=model,
                       input_config=config,
                       train_iter=train_iter,
                       test_iter=test_iter,
                       val_iter=val_iter,
                       input_log=log)
    # training
    # train_func.train()

    # testing
    train_func.evaluate(test=True)

    train_func.case_analysis()

    log.close()
