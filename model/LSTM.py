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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.dataset_class import *
from src.utils import *
from metric import metric


class LSTMConfig(object):
    def __init__(self):
        super(LSTMConfig, self).__init__()
        self.ds = TextDataset()
        self.model_name = 'LSTM'
        self.embed_dim = self.ds.embed_dim
        self.hidden_size = 32
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 128
        self.max_doc_len = self.ds.max_doc_len
        self.top_k_list = [1, 5, 10, 15, 20, 25, 30]
        self.lr = 0.001
        self.dropout = 0.2
        self.num_layer = 1
        self.num_category = self.ds.num_category
        self.num_mashup = self.ds.num_mashup
        self.num_api = self.ds.num_api
        self.vocab_size = self.ds.vocab_size
        self.embed = self.ds.embed


class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        if config.embed is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embed, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.vocab_size - 1)

        self.lstm = nn.LSTM(input_size=config.embed_dim,
                            hidden_size=config.hidden_size,
                            num_layers=config.num_layer,
                            bidirectional=True,
                            batch_first=True)
        self.tanh = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.fc = nn.Linear(config.hidden_size*2, config.num_api)

    def forward(self, x, lengths):
        embed = self.embedding(x)  # [batch_size, seq_len, embed_size]
        sorted_lengths, indices = torch.sort(lengths, descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)
        sorted_embed = embed[indices]
        packed = pack_padded_sequence(sorted_embed, sorted_lengths, batch_first=True)
        H, _ = self.lstm(packed)  # [batch_size, seq_len, hidden_size * num_direction]
        H, _ = pad_packed_sequence(H, batch_first=True)

        desorted_H = H[desorted_indices]
        M = self.tanh(desorted_H)  # [batch_size, seq_len, hidden_size * num_direction]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [batch_size, seq_len, 1]
        out = H * alpha  # [batch_size, seq_len, hidden_size * num_direction]
        out = torch.sum(out, dim=1)  # [batch_size, hidden_size * num_direction]
        # relu = torch.relu(out)
        fc = self.fc(out)        # category_out = self.fc(out)  # [batch_size, num_category]
        return torch.sigmoid(fc)


class Train(object):
    def __init__(self, model, config, train_iter, test_iter, val_iter, case_iter, log, model_path=None):
        self.config = config
        self.model = model.to(config.device)
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.val_iter = val_iter
        self.case_iter = case_iter
        self.criterion = torch.nn.BCELoss()
        self.optim = torch.optim.Adam(model.parameters(), lr=config.lr)
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

            for batch_idx, batch_data in enumerate(self.train_iter):
                # batch_data = [index, des, category_performance, used_api]
                self.optim.zero_grad()
                api_target = batch_data[3].float().to(self.config.device)
                input_data = batch_data[1].to(self.config.device)
                des_lens = batch_data[4].to(self.config.device)
                api_pred = self.model(input_data, des_lens)
                api_loss_ = self.criterion(api_pred, api_target)
                api_loss_.backward()
                api_loss.append(api_loss_.item())
                self.optim.step()

            api_loss = np.average(api_loss)
            info = '[Epoch:%s] ApiLoss:%s\n' % (epoch+1, api_loss)
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

        ndcg_a = np.zeros(len(self.top_k_list))
        recall_a = np.zeros(len(self.top_k_list))
        ap_a = np.zeros(len(self.top_k_list))
        pre_a = np.zeros(len(self.top_k_list))
        api_loss = []
        num_batch = len(data_iter)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_iter):
                api_target = batch_data[3].float().to(self.config.device)
                input_data = batch_data[1].to(self.config.device)
                des_lens = batch_data[4].to(self.config.device)
                api_pred = self.model(input_data, des_lens)
                api_loss_ = self.criterion(api_pred, api_target)
                api_loss.append(api_loss_.item())
                api_pred = api_pred.cpu().detach()
                ndcg_, recall_, ap_, pre_ = metric(batch_data[3], api_pred, top_k_list=self.top_k_list)
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
        case_path = 'case/{0}.json'.format(config.model_name)
        a_case = open(case_path, mode='w')
        api_case = []
        self.model.eval()
        with torch.no_grad():

            for batch_idx, batch_data in enumerate(self.case_iter):
                index = batch_data[0].to(self.config.device)
                des = batch_data[1].to(self.config.device)
                des_len = batch_data[4].to(self.config.device)
                api_target = []
                for api_data in batch_data[3]:
                    if isinstance(api_data.nonzero().squeeze().tolist(), list):
                        api_target.append(api_data.nonzero().squeeze().tolist())
                    else:
                        api_target.append([api_data.nonzero().squeeze().tolist()])
                api_pred_ = self.model(des, des_len)
                api_pred = api_pred_.cpu().argsort(descending=True)[:, :5].tolist()
                for i, api_tuple in enumerate(zip(api_target, api_pred)):
                    target = []
                    pred = []
                    name = self.config.ds.mashup_ds.name[index[i].cpu().tolist()]
                    for t in api_tuple[0]:
                        target.append(self.config.ds.mashup_ds.used_api_mlb.classes_[t])
                    for t in api_tuple[1]:
                        pred.append(self.config.ds.mashup_ds.used_api_mlb.classes_[t])
                    api_case.append((name, target, pred))
        json.dump(api_case, a_case)
        a_case.close()



if __name__ == '__main__':
    # load ds
    print('Start ...')
    now = time.time()
    config = LSTMConfig()
    ds = config.ds
    model = LSTM(config)
    print('Time for loading model: ', get_time(now))
    now = int(time.time())
    timeStruct = time.localtime(now)
    strTime = time.strftime("%Y-%m-%d", timeStruct)
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
                       log=log)
    # train_func.train()

    train_func.evaluate(test=True)
    train_func.case_analysis()

    log.close()

