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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation



class CFConfig(object):
    def __init__(self):
        super(CFConfig, self).__init__()
        self.ds = MashupDataset()
        self.api = ApiDataset()
        self.model_name = 'CF'
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 128
        self.num_mashup = len(self.ds.name)
        self.num_api = self.ds.num_api
        self.num_category = self.ds.num_category
        self.feature_size = 16
        self.m2a_mat = self.get_m2a()
        cntVector = CountVectorizer()
        cntTf = cntVector.fit_transform([' '.join(des) for des in self.ds.description])
        lda = LatentDirichletAllocation(n_components=self.feature_size,
                                        random_state=2020)
        self.docres = torch.tensor(lda.fit_transform(cntTf))
        self.m2m_sim_mat = self.get_m2m_sim_mat()
        self.m2a_sim_mat = torch.mm(self.m2m_sim_mat, self.m2a_mat.double())

    def get_m2a(self):
        matrix = torch.zeros(len(self.ds), self.ds.num_api)
        for i, api in enumerate(self.ds.used_api):
            matrix[i] = torch.tensor(self.ds.used_api_mlb.transform([api])).squeeze()
        return matrix

    def get_m2m_sim_mat(self):
        matrix = torch.mm(self.docres, self.docres.permute(1, 0))
        for i in range(matrix.size(0)):
            for j in range(matrix.size(1)):
                if i == j:
                    matrix[i, j] = torch.tensor(0)
        return matrix


class CF(torch.nn.Module):
    def __init__(self, config):
        super(CF, self).__init__()
        self.m2a_mat = nn.Embedding.from_pretrained(config.m2a_sim_mat.float(), freeze=True)
        self.norm = nn.BatchNorm1d(config.num_api)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices):
        embed = self.m2a_mat(user_indices)
        # norm = self.norm(embed)
        return self.logistic(embed)


class Train(object):
    def __init__(self, model, config, train_iter, test_iter, val_iter, case_iter, log, input_ds, model_path=None):
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
        self.ds = input_ds
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = 'checkpoint/%s.pth' % self.config.model_name
        self.early_stopping = EarlyStopping(patience=7, path=self.model_path)

    def train(self):
        print('Start training ...')
        self.model.train()
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
                api_target = batch_data[3].float().to(self.config.device)
                api_pred = self.model(batch_data[0].to(self.config.device))
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
            self.model.eval()
            for batch_idx, batch_data in enumerate(self.case_iter):
                index = batch_data[0].to(self.config.device)
                des = batch_data[1].to(self.config.device)
                api_target = []
                for api_data in batch_data[3]:
                    if isinstance(api_data.nonzero().squeeze().tolist(), list):
                        api_target.append(api_data.nonzero().squeeze().tolist())
                    else:
                        api_target.append([api_data.nonzero().squeeze().tolist()])
                api_pred_ = self.model(index)
                api_pred_ = api_pred_.cpu().argsort(descending=True)[:, :5].tolist()
                for i, api_tuple in enumerate(zip(api_target, api_pred_)):
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
    now = time.time()
    # load ds
    print('Start ...')
    ds = TextDataset()
    print('Time for loading dataset: ', get_time(now))
    now = time.time()

    config = CFConfig()
    model = CF(config)
    print('Time for loading model: ', get_time(now))

    now = int(time.time())
    timeStruct = time.localtime(now)
    strTime = time.strftime("%Y-%m-%d-%H:%M:%S", timeStruct)
    log_path = './log/%s.log' % config.model_name
    log = open(log_path, mode='a')
    log.write(strTime)

    train_idx, val_idx, test_idx = get_indices(ds.mashup_ds)
    train_iter = DataLoader(ds.mashup_ds, batch_size=config.batch_size, sampler=SubsetRandomSampler(train_idx))
    val_iter = DataLoader(ds.mashup_ds, batch_size=len(val_idx), sampler=SubsetRandomSampler(val_idx))
    test_iter = DataLoader(ds.mashup_ds, batch_size=len(test_idx), sampler=SubsetRandomSampler(test_idx))
    case_iter = DataLoader(ds.mashup_ds, batch_size=len(ds.mashup_ds))

    model_path = 'checkpoint/%s.pth' % config.model_name
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    train_func = Train(model=model,
                       config=config,
                       train_iter=train_iter,
                       test_iter=test_iter,
                       val_iter=val_iter,
                       case_iter=case_iter,
                       log=log,
                       input_ds=ds)
    # train_func.train()

    train_func.evaluate(test=True)
    train_func.case_analysis()

    log.close()

