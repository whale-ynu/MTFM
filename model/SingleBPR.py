#%%
import os
import sys
import torch
import time
curPath = os.path.abspath(os.path.dirname('__file__'))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.dataset_class import *
from src.utils import *
from metric import metric


class BPRConfig(object):
    def __init__(self, ds_config):
        self.ds = TextDataset()
        self.model_name = 'SingleBPR'
        self.neg_num = 20  # negative sampling
        self.max_doc_len = ds_config.max_doc_len
        self.weight_decay = 0.075
        self.feature_size = 4
        self.num_category = ds_config.num_category
        self.num_mashup = ds_config.num_mashup
        self.num_api = ds_config.num_api
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = 0.2
        self.batch_size = 128
        self.lr = 1e-3
        self.ds = ds_config


class BPR(nn.Module):
    def __init__(self, config):
        super(BPR, self).__init__()
        self.W = nn.Embedding(config.num_mashup, config.feature_size)
        self.H = nn.Embedding(config.num_api, config.feature_size)
        self.init_weights()
        self.weight_decay = config.weight_decay
        self.dropout = nn.Dropout(p=config.dropout)
        self.tanh = nn.Tanh()
        self.logsigmoid = nn.LogSigmoid()

    def init_weights(self):
        nn.init.kaiming_normal_(self.W.weight)
        nn.init.kaiming_normal_(self.H.weight)

    def forward(self, u, i, j):
        u_bpr = self.W(u)  # [batch_size, feature_size]
        i_bpr = self.H(i)
        j_bpr = self.H(j)
        x_ui = torch.mul(u_bpr, i_bpr).sum(dim=1)
        x_uj = torch.mul(u_bpr, j_bpr).sum(dim=1)
        x_uij = x_ui - x_uj
        log_prob = self.logsigmoid(x_uij).sum()
        regularization = self.weight_decay * (u_bpr.norm(dim=1).pow(2).sum() + i_bpr.norm(dim=1).pow(2).sum() + j_bpr.norm(dim=1).pow(2).sum())
        return -log_prob + regularization

    def recommend(self, u):
        with torch.no_grad():
            bpr_u = self.W(u)
            u_bpr = torch.mm(bpr_u, self.H.weight.t())
        return torch.sigmoid(u_bpr)


class Train(object):
    def __init__(self, input_model, input_config, train_iter, test_iter, val_iter, case_iter, input_log, model_path=None):
        self.model = input_model
        self.config = input_config
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.val_iter = val_iter
        self.case_iter = case_iter
        self.criterion = torch.nn.BCELoss()
        self.optim = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        self.epoch = 7
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
            self.model.train()
            api_loss = []

            for batch_idx, batch_data in enumerate(self.train_iter):
                # batch_data = [batch_size, mashup, api_i, api_j]
                self.optim.zero_grad()
                bpr_loss_ = self.model(batch_data[0][0].to(self.config.device),
                              batch_data[1][0].to(self.config.device),
                              batch_data[2][0].to(self.config.device))
                bpr_loss_.backward()
                self.optim.step()
                api_loss.append(bpr_loss_.item())

            api_loss = np.average(api_loss)

            info = '[Epoch:%s] ApiLoss:%s' % (epoch+1, api_loss.round(6))
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

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_iter):
                api_loss_ = self.model(batch_data[0][0].to(self.config.device),
                              batch_data[1][0].to(self.config.device),
                              batch_data[2][0].to(self.config.device))
                api_loss.append(api_loss_.item())
                api_pred = self.model.recommend(batch_data[0][0].to(self.config.device))
                api_pred = api_pred.cpu().detach()

                ndcg_, recall_, ap_, pre_ = metric(batch_data[0][3], api_pred, top_k_list=self.top_k_list)
                ndcg_a += ndcg_
                recall_a += recall_
                ap_a += ap_
                pre_a += pre_

        api_loss = np.average(api_loss)

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
            self.model.eval()
            for batch_idx, batch_data in enumerate(self.case_iter):
                index = batch_data[0].to(self.config.device)
                api_target = []
                for api_data in batch_data[3]:
                    if isinstance(api_data.nonzero().squeeze().tolist(), list):
                        api_target.append(api_data.nonzero().squeeze().tolist())
                    else:
                        api_target.append([api_data.nonzero().squeeze().tolist()])
                api_pred_ = self.model.recommend(index)
                api_pred = api_pred_.cpu().argsort(descending=True)[:, :5].tolist()
                for i, api_tuple in enumerate(zip(api_target, api_pred)):
                    target = []
                    pred = []
                    name = self.config.ds.mashup_ds.name[index[i].cpu().tolist()]
                    for t in api_tuple[0]:
                        target.append(self.config.ds.mashup_ds.used_api_mlb.classes_[t])
                    for t in api_tuple[1]:
                        pred.append(self.config.ds.mashup_ds.used_api_mlb.classes_[t])
                    mashup_case.append((name, target, pred))
        json.dump(mashup_case, case)
        case.close()


if __name__ == '__main__':
    tmp_time = time.time()
    # load ds
    print('Start ...')
    ds = TextDataset()
    print('Time for loading dataset: ', get_time(tmp_time))
    tmp_time = time.time()

    model_config = BPRConfig(ds_config=ds)
    model = BPR(config=model_config)
    model.to(model_config.device)
    log_path = './log/%s.log' % model_config.model_name
    log = open(log_path, mode='a')
    now = int(time.time())
    timeStruct = time.localtime(now)
    strTime = time.strftime("%Y-%m-%d", timeStruct)
    log.write('{0} {1}\n'.format(model_config.model_name, strTime))
    train_idx, val_idx, test_idx = get_indices(ds.mashup_ds)
    train_ds = BPRDataset(sample_indices=list(range(len(ds.mashup_ds))), neg_num=model_config.neg_num)

    train_iter = DataLoader(train_ds, batch_size=model_config.batch_size, shuffle=True)

    val_ds = BPRDataset(sample_indices=val_idx, neg_num=1)

    val_iter = DataLoader(val_ds, batch_size=len(val_ds))

    test_ds = BPRDataset(sample_indices=test_idx, neg_num=1)

    test_iter = DataLoader(test_ds, batch_size=len(test_ds))

    case_iter = DataLoader(ds.mashup_ds, batch_size=len(ds.mashup_ds))

    model_path = 'checkpoint/%s.pth' % model_config.model_name
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    train_func = Train(input_model=model,
                       input_config=model_config,
                       train_iter=train_iter,
                       test_iter=test_iter,
                       val_iter=val_iter,
                       case_iter=case_iter,
                       input_log=log)
    # train_func.train()
    train_func.evaluate(test=True)
    train_func.case_analysis()

    log.close()
