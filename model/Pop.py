"""
It is the most commonly used baseline model in the field of recommender systems.
For each tag or category_performance, a queue of APIs is created through sorting APIs by their
frequencies in the mashup developments. For each tag or category_performance ofQ, one API is
selected to enter into the recommendation list according to its queue order in
each iteration, and such a process repeats several times until top-N candidates
are completed.
"""

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
from torch.utils.data import DataLoader, SubsetRandomSampler
from src.dataset_class import *
from src.utils import *
from metric import metric


class PopConfig(object):
    def __init__(self):
        super(PopConfig, self).__init__()
        self.api_ds = ApiDataset()
        self.mashup_ds = MashupDataset()
        self.model_name = 'Pop'
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 128
        self.pop_weight = 0.25
        self.category_freq, self.api_freq = self.get_freq_matrix()
        self.c2a_mat = self.get_c2a_matrix()
        self.m2c_mat = self.get_m2c_matrix()
        self.freq_mat = F.normalize(self.category_freq.t() * self.api_freq)
        self.m2a_mat = torch.mm(self.m2c_mat, torch.mul(self.c2a_mat, self.freq_mat) + F.normalize(self.api_freq)*self.pop_weight)

    def get_freq_matrix(self):
        category_freq = torch.zeros(1, len(self.mashup_ds.category_mlb.classes_))
        api_freq = torch.zeros(1, len(self.mashup_ds.used_api_mlb.classes_))
        for category in self.mashup_ds.category:
            category_freq += torch.tensor(self.mashup_ds.category_mlb.transform([category])).squeeze()
        for api in self.mashup_ds.used_api:
            api_freq += torch.tensor(self.mashup_ds.used_api_mlb.transform([api])).squeeze()
        return category_freq, api_freq

    def get_c2a_matrix(self):
        matrix = torch.zeros(len(self.api_ds), self.api_ds.num_category)
        for i, category in enumerate(self.api_ds.category):
            matrix[i] = torch.tensor(self.api_ds.category_mlb.transform([category])).squeeze()
        return matrix.t()

    def get_m2c_matrix(self):
        matrix = torch.zeros(len(self.mashup_ds), self.mashup_ds.num_category)
        for i, category in enumerate(self.mashup_ds.category):
            matrix[i] = torch.tensor(self.mashup_ds.category_mlb.transform([category])).squeeze()
        return matrix


class Pop(nn.Module):
    def __init__(self, config):
        super(Pop, self).__init__()
        self.embed = nn.Embedding.from_pretrained(config.m2a_mat.float(), freeze=True)
        self.norm = nn.BatchNorm1d(config.mashup_ds.num_api)
        self.logistic = nn.Sigmoid()

    def forward(self, u):
        embed = self.embed(u)
        # norm = self.norm(embed)
        return self.logistic(embed)


class Train(object):
    def __init__(self, model, config, train_iter, test_iter, val_iter, case_iter, log, model_path=None):
        self.config = config
        self.model = model.to(config.device)
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.val_iter = val_iter
        self.case_iter = case_iter
        self.criterion = torch.nn.BCELoss()
        self.epoch = 1
        self.top_k_list = [1, 5, 10, 15, 20, 25, 30]
        self.log = log
        self.early_stopping = EarlyStopping(patience=7, path=model_path)
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = 'checkpoint/%s.pth' % self.config.model_name
        self.early_stopping = EarlyStopping(patience=7, path=self.model_path)

    def train(self):
        print('Start training ...')
        self.model.eval()
        for epoch in range(self.epoch):
            api_loss = []
            for batch_idx, batch_data in enumerate(self.train_iter):
                # batch_data = [index, des, category_performance, used_api]
                api_target = batch_data[3].float().to(self.config.device)
                input_data = batch_data[0].to(self.config.device)
                api_pred = self.model(input_data)
                api_loss_ = self.criterion(api_pred, api_target)
                api_loss.append(api_loss_.item())

            api_loss = np.average(api_loss)
            info = '[Epoch:%s] ApiLoss:%s\n' % (epoch+1, api_loss.round(6))
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
        num_batch = len(data_iter)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_iter):
                index = batch_data[0].to(self.config.device)
                des = batch_data[1].to(self.config.device)
                category_target = batch_data[2].float().to(self.config.device)
                api_target = batch_data[3].float().to(self.config.device)

                api_pred = self.model(index)
                api_loss_ = self.criterion(api_pred, api_target)
                api_loss.append(api_loss_.item())

                api_pred = api_pred.cpu().detach()

                ndcg_, recall_, ap_, pre_ = metric(batch_data[3], api_pred.cpu(),
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
            for batch_idx, batch_data in enumerate(self.case_iter):
                index = batch_data[0].to(self.config.device)
                des = batch_data[1].to(self.config.device)
                api_target = []
                for api_data in batch_data[3]:
                    if isinstance(api_data.nonzero().squeeze().tolist(), list):
                        api_target.append(api_data.nonzero().squeeze().tolist())
                    else:
                        api_target.append([api_data.nonzero().squeeze().tolist()])
                api_pred_= self.model(index)
                api_pred = api_pred_.cpu().argsort(descending=True)[:, :5].tolist()
                for i, api_tuple in enumerate(zip(api_target, api_pred)):
                    target = []
                    pred = []
                    name = self.config.mashup_ds.name[index[i].cpu().tolist()]
                    for t in api_tuple[0]:
                        target.append(self.config.mashup_ds.used_api_mlb.classes_[t])
                    for t in api_tuple[1]:
                        pred.append(self.config.mashup_ds.used_api_mlb.classes_[t])
                    mashup_case.append((name, target, pred))

        json.dump(mashup_case, case)
        case.close()


if __name__ == '__main__':
    tmp_time = time.time()
    # load ds
    print('Start ...')
    mashup_ds = MashupDataset()
    api_ds = ApiDataset()
    ds = TextDataset()
    print('Time for loading dataset: ', get_time(tmp_time))
    tmp_time = time.time()

    pop_config = PopConfig()
    pop_model = Pop(pop_config)
    now = int(time.time())
    timeStruct = time.localtime(now)
    strTime = time.strftime("%Y-%m-%d", timeStruct)
    log_path = './log/%s.log' % pop_config.model_name
    log = open(log_path, mode='a')
    log.write('{0} {1}\n'.format(pop_config.model_name, strTime))

    train_idx, val_idx, test_idx = get_indices(mashup_ds)

    train_iter = DataLoader(ds.mashup_ds, batch_size=pop_config.batch_size, sampler=SubsetRandomSampler(train_idx))
    test_iter = DataLoader(ds.mashup_ds, batch_size=len(test_idx), sampler=SubsetRandomSampler(test_idx))
    val_iter = DataLoader(ds.mashup_ds, batch_size=len(val_idx), sampler=SubsetRandomSampler(val_idx))
    case_iter = DataLoader(ds.mashup_ds, batch_size=len(ds.mashup_ds))

    model_path = 'checkpoint/%s.pth' % pop_config.model_name
    pop_model.load_state_dict(torch.load(model_path, map_location=pop_config.device))

    train_func = Train(model=pop_model,
                       config=pop_config,
                       train_iter=train_iter,
                       test_iter=test_iter,
                       val_iter=val_iter,
                       case_iter=case_iter,
                       log=log)
    # train_func.train()
    train_func.evaluate(test=True)
    train_func.case_analysis()
    log.close()
