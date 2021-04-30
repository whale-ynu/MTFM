# coding=utf-8
import torch
import datetime
import os
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
tokenizer = RegexpTokenizer("[\s,'\.]", gaps=True)
from nltk.stem import WordNetLemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
from string import punctuation
import re
import math
import time
from datetime import timedelta
from tqdm import tqdm
import numpy as np
import random
random.seed(2020)
from random import shuffle


def get_indices(dataset):
    data_idx = list(range(len(dataset)))
    shuffle(data_idx)
    split_num = int(len(dataset) / 10)
    val_idx = data_idx[:split_num]
    test_idx = data_idx[split_num:split_num * 2]
    train_idx = data_idx[split_num * 2:]
    return train_idx, val_idx, test_idx


def get_time(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def tokenize(text):
    """
    tokenize
    :param input_str:
    :return:
    """
    if text == '' or text is None:
        return []
    text = text.lower().replace('\n', ' ')
    # 缩写替换
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"cannot", "can not ", text)
    text = re.sub(r"what\'s", "what is", text)
    text = re.sub(r"What\'s", "what is", text)
    text = re.sub(r"\'ve ", " have ", text)
    text = re.sub(r"n\'t", " not ", text)
    text = re.sub(r"i\'m", "i am ", text)
    text = re.sub(r"I\'m", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub('[%s]' % re.escape(punctuation), "", text)
    #
    r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
    text = re.sub(r4, ' ', text)

    tokens = tokenizer.tokenize(text)
    # 词性还原
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # print(tokens)
    # 词干提取
    # tokens = [stemmer.stem(word) for word in tokens]
    # print(tokens)
    # 去除停用词
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return tokens


def resume_checkpoint(model, model_dir, device_id):
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            torch.save(model.state_dict(), self.path)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


