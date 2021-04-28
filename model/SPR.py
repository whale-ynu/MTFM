"""
Web Service Recommendation With Reconstructed Profile From Mashup Descriptions
"""

#%%
import os
import sys
curPath = os.path.abspath(os.path.dirname('__file__'))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from src.dataset_class import *
from src.utils import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim import corpora, similarities
from gensim.models import AuthorTopicModel
from metric import metric


class SPRConfig(object):
    def __init__(self):
        super(SPRConfig, self).__init__()
        self.ds = TextDataset()
        self.mashup_ds = MashupDataset()
        self.api_ds = ApiDataset()
        self.model_name = 'SPR'
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 128
        self.num_mashup = len(self.mashup_ds)
        self.num_api = len(self.api_ds)
        self.feature_size = 8
        self.lr = 1e-3
        self.h = 1  # Thresholds for dominant words generation
        self.g = 0.5  # Thresholds for dominant words generation
        self.dictionary = corpora.Dictionary(self.mashup_ds.description)
        author2doc = {}
        self.doc_term_matrix = [self.dictionary.doc2bow(doc) for doc in self.mashup_ds.description]
        for i, api in enumerate(self.mashup_ds.used_api_mlb.classes_):
            author2doc[api] = []
        for j, used_api in enumerate(self.mashup_ds.used_api):
            for tmp_api in used_api:
                author2doc[tmp_api].append(j)
        self.ATM = AuthorTopicModel(self.doc_term_matrix, author2doc=author2doc, id2word=self.dictionary, num_topics=self.feature_size)
        self.M2D = np.zeros((len(self.doc_term_matrix), len(self.dictionary)))
        for i, s in enumerate(self.doc_term_matrix):
            for w in s:
                self.M2D[i, w[0]] = 0.1
        self.RSP = np.zeros((self.mashup_ds.num_api, len(self.dictionary)))
        for i, docid in enumerate(self.ATM.author2doc.values()):
            for doc in docid:
                self.RSP[i] += self.M2D[doc]*0.2
        self.RSP += 0.5
        self.w_sum = self.RSP.sum(axis=0)
        self.P = self.RSP/self.w_sum
        self.DW = np.zeros((self.mashup_ds.num_api, len(self.dictionary)))
        for s in range(self.mashup_ds.num_api):
            for w in range(len(self.dictionary)):
                if self.RSP[s, w] >= self.h and self.P[s, w] >= self.g:
                    self.DW[s, w] = 0.1

        self.L = []
        for des in self.mashup_ds.description:
            self.L.append(len(des))
        self.R = np.zeros((self.num_mashup, self.num_api))
        """
        For each service s in S:
            Calculate relevance score r (s, Q) by equation (3)
            For each word w in Q:
                If (s,w) ∈ DW:
                    r(s, Q) = r(s, Q) + L
            End
        End
        """
        # for i, m2w in enumerate(self.M2D):
        #     for j, s2w in enumerate(self.DW):
        #         self.R[i, j] += np.dot(m2w, s2w)*self.L[i]
        self.m2d = torch.tensor(self.M2D)
        self.a2d = torch.tensor(self.P)
        self.R = F.normalize(torch.tensor(self.R))


class SPR(torch.nn.Module):
    def __init__(self, config):
        super(SPR, self).__init__()
        self.m2d_mat = nn.Embedding.from_pretrained(config.m2d.float(), freeze=False)
        self.a2d_mat = nn.Embedding.from_pretrained(config.a2d.float(), freeze=False)
        self.R = nn.Embedding.from_pretrained(config.R.float(), freeze=False)
        self.logitic = nn.Softmax()

    def forward(self, user_indices):
        m2d = self.m2d_mat(user_indices)
        a2d = self.a2d_mat.weight.t()
        r = F.normalize(torch.mm(m2d, a2d))
        r += self.R(user_indices)
        return self.logitic(r)


class Train(object):
    def __init__(self, input_model, config, train_iter, test_iter, val_iter, case_iter, log, model_path=None):
        self.config = config
        self.model = input_model.to(config.device)
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.val_iter = val_iter
        self.case_iter = case_iter
        self.criterion = torch.nn.BCELoss()
        self.optim = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.epoch = 1
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
                    name = self.config.mashup_ds.name[index[i].cpu().tolist()]
                    for t in api_tuple[0]:
                        target.append(self.config.mashup_ds.used_api_mlb.classes_[t])
                    for t in api_tuple[1]:
                        pred.append(self.config.mashup_ds.used_api_mlb.classes_[t])
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
    config = SPRConfig()
    model = SPR(config)
    print('Time for loading model: ', get_time(now))

    now = int(time.time())
    timeStruct = time.localtime(now)
    strTime = time.strftime("%Y-%m-%d-%H:%M:%S", timeStruct)
    log_path = './log/%s.log' % config.model_name
    log = open(log_path, mode='a')
    log.write('{0} {1} batch_size:{2}\n'.format(config.model_name, strTime, str(config.batch_size)))

    train_idx, val_idx, test_idx = get_indices(ds.mashup_ds)
    train_iter = DataLoader(ds.mashup_ds, batch_size=config.batch_size, sampler=SubsetRandomSampler(train_idx))
    test_iter = DataLoader(ds.mashup_ds, batch_size=len(test_idx), sampler=SubsetRandomSampler(test_idx))
    val_iter = DataLoader(ds.mashup_ds, batch_size=len(val_idx), sampler=SubsetRandomSampler(val_idx))
    case_iter = DataLoader(ds.mashup_ds, batch_size=len(ds.mashup_ds))

    model_path = 'checkpoint/%s.pth' % config.model_name
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    train_func = Train(input_model=model,
                       config=config,
                       train_iter=train_iter,
                       test_iter=test_iter,
                       val_iter=val_iter,
                       case_iter=case_iter,
                       log=log)

    # train_func.train()
    train_func.evaluate(test=True)
    train_func.case_analysis()
    log.close()


