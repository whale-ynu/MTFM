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
from src.dataset_class import *
from src.utils import *
from metric import precision, recall, ndcg, ap
from scipy.sparse import csr_matrix
from fast_pagerank import pagerank_power


class RWR(object):

    def __init__(self, mashup_ds, api_ds, train_idx, input_log):
        # a dictionary to map string to identifier
        self.id_factory = {}
        self.num_api = len(api_ds.name)
        self.num_mashup = len(mashup_ds.name)
        self.log = input_log
        self.top_k_list = [1, 5, 10, 15, 20, 25, 30]
        count = 0
        # assign identifier for api
        for str in api_ds.name:
            self.id_factory['A#' + str] = count  # marked with prefix 'A#'
            count += 1
        
        # assign identifier for category
        for i in range(len(api_ds.category)):
            for cate in  api_ds.category[i]:
                prefix_cat = 'C#' + cate  # marked with prefix 'C#'
                if prefix_cat not in self.id_factory:
                    
                    self.id_factory[prefix_cat] = count
                    count += 1

        for i in range(len(mashup_ds.category)):
            for cate in mashup_ds.category[i]:
                prefix_cat = 'C#' + cate  # marked with prefix 'C#'
                if prefix_cat not in self.id_factory:
                    self.id_factory[prefix_cat] = count
                    count += 1
        # assign identifier for mashup            
        for str in mashup_ds.name:
            self.id_factory['M#' + str] = count  # marked with prefix 'M#'
            count += 1
        # create knowledge graph by adding links  
        link_source = []
        link_target = []
        link_weight = []
        
        for idx in train_idx:
            # adding links between mashups and categories
            new_mashup_idx = self.id_factory['M#' + mashup_ds.name[idx]]
            for cat in mashup_ds.category[idx]:
                new_cat_idx = self.id_factory['C#' + cat]
                link_source.append(new_mashup_idx)
                link_target.append(new_cat_idx)
                link_weight.append(1)
                
                link_target.append(new_mashup_idx)
                link_source.append(new_cat_idx)
                link_weight.append(1)
            # adding links between mashups and apis    
            for api in mashup_ds.used_api[idx]:
                new_api_idx = self.id_factory['A#' + api]
                link_source.append(new_mashup_idx)
                link_target.append(new_api_idx)
                link_weight.append(1)
                
                link_target.append(new_mashup_idx)
                link_source.append(new_api_idx)
                link_weight.append(1)
                
        # adding links between apis and categories
        for i in range(len(api_ds.category)):
            new_api_idx = self.id_factory['A#' + api_ds.name[i]]
            for cat in api_ds.category[i]:
                new_cat_idx = self.id_factory['C#' + cat]
                link_source.append(new_api_idx)
                link_target.append(new_cat_idx)
                link_weight.append(1)
                
                link_target.append(new_api_idx)
                link_source.append(new_cat_idx)
                link_weight.append(1)
        
        self.G = csr_matrix((link_weight, (link_source, link_target)),
                     shape=(len(self.id_factory), len(self.id_factory)))            
                
    def evaluate(self, test_idx):
        print('Start testing ...')
        # API
        ndcg_a = np.zeros(len(self.top_k_list))
        recall_a = np.zeros(len(self.top_k_list))
        ap_a = np.zeros(len(self.top_k_list))
        pre_a = np.zeros(len(self.top_k_list))
        
        personalize_vector = np.zeros(self.G.shape[0])
        
        for idx in tqdm (test_idx, desc="Perform rand walk and make recommendation"):
            new_mashup_idx = self.id_factory['M#' + mashup_ds.name[idx]]
            # initialize G and personalize_vector
            personalize_vector[new_mashup_idx] = 1.0
            for cat in mashup_ds.category[idx]:
                new_cat_idx = self.id_factory['C#' + cat]
                self.G[new_mashup_idx, new_cat_idx] = 1
            
            rwr = pagerank_power(self.G, p=0.85, personalize=personalize_vector, tol=1e-6)
            #recover links between testing mashups and categories
            for cat in mashup_ds.category[idx]:
                new_cat_idx = self.id_factory['C#' + cat]
                self.G[new_mashup_idx, new_cat_idx] = 0
            # recover original G    
            self.G.eliminate_zeros()
            personalize_vector[new_mashup_idx] = 0

            # build TOP_N ranking list         
            ranklist = sorted(zip(rwr[0:self.num_api], api_ds.name), reverse=True)
            #print(ranklist)
            
            for n in range(len(self.top_k_list)):
                sublist = ranklist[:self.top_k_list[n]]
                score, pred = zip(*sublist)
                p_at_k = precision(mashup_ds.used_api[idx], pred)
                r_at_k = recall(mashup_ds.used_api[idx], pred)
                ndcg_at_k = ndcg(mashup_ds.used_api[idx], pred)
                ap_at_k = ap(mashup_ds.used_api[idx], pred)
                
                pre_a[n] += p_at_k
                recall_a[n] += r_at_k
                ndcg_a[n] += ndcg_at_k
                ap_a[n] += ap_at_k
            
        # calculate the final scores of metrics      
        for n in range(len(self.top_k_list)):
            pre_a[n] /= len(test_idx)
            recall_a[n] /= len(test_idx)
            ndcg_a[n] /= len(test_idx)
            ap_a[n] /= len(test_idx)

        info = '[#Test %d]\n'\
               'NDCG_A:%s\n' \
               'AP_A:%s\n' \
               'Pre_A:%s\n' \
               'Recall_A:%s\n' \
                % (len(test_idx), ndcg_a.round(6), ap_a.round(6), pre_a.round(6), recall_a.round(6))

        print(info)
        log.write(info)
        log.flush()
        #=======================================================================
        # self.log.write(info + '\n')
        # self.log.flush()
        #=======================================================================

#===============================================================================
    def case_analysis(self, case_index):
        case_path = 'case/{0}.json'.format('RWR')
        case = open(case_path, mode='w')
        api_case = []
        personalize_vector = np.zeros(self.G.shape[0])

        for idx in tqdm(case_index, desc="Perform rand walk and make recommendation"):
            new_mashup_idx = self.id_factory['M#' + mashup_ds.name[idx]]
            # initialize G and personalize_vector
            personalize_vector[new_mashup_idx] = 1.0
            for cat in mashup_ds.category[idx]:
                new_cat_idx = self.id_factory['C#' + cat]
                self.G[new_mashup_idx, new_cat_idx] = 1

            rwr = pagerank_power(self.G, p=0.85, personalize=personalize_vector, tol=1e-6)
            # recover links between testing mashups and categories
            for cat in mashup_ds.category[idx]:
                new_cat_idx = self.id_factory['C#' + cat]
                self.G[new_mashup_idx, new_cat_idx] = 0
            # recover original G
            self.G.eliminate_zeros()
            personalize_vector[new_mashup_idx] = 0

            # build TOP_N ranking list
            ranklist = sorted(zip(rwr[0:self.num_api], api_ds.name), reverse=True)
            # print(ranklist)


            api_case.append([mashup_ds.name[idx], mashup_ds.used_api[idx], ranklist[:5]])
        json.dump(api_case, case)
        case.close()

#===============================================================================


if __name__ == '__main__':
    # load ds
    print('Start ...')
    start_time = time.time()
    now = time.time()
    mashup_ds = MashupDataset()
    api_ds = ApiDataset()
    
    print('Time for loading dataset: ', get_time(now))
 
    # initial
    log_path = 'log/{0}.log'.format('RWR')
    log = open(log_path, mode='a')
    train_idx, val_idx, test_idx = get_indices(mashup_ds)
    rwr = RWR(mashup_ds, api_ds, train_idx, log)
    print(len(train_idx))
    print(len(test_idx))
    rwr.evaluate(test_idx)
    rwr.case_analysis(case_index=list(range(len(mashup_ds))))
    # training
#===============================================================================
#     now = int(time.time())
#     timeStruct = time.localtime(now)
#     strTime = time.strftime("%Y-%m-%d", timeStruct)

#     log.write(strTime)
#     log.flush()
# 
#     # training
#     # train_func.train()
# 
#     # testing
#     train_func.evaluate(test=True)
# 
#     train_func.case_analysis()
    log.close()
#===============================================================================
