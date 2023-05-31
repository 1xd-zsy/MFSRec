# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import os
import sys

from model.mashup_api_dataset import FCMashupAPI
from tools.metric import *

curPath = os.path.abspath(os.path.dirname('__file__'))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from scipy.sparse import csr_matrix
from fast_pagerank import pagerank_power
from tqdm import tqdm


class RWR(object):
    def __init__(self, mashup_api_ds):
        # a dictionary to map string to identifier
        self.id_factory = {}
        self.num_api = mashup_api_ds.api_num
        self.num_mashup = mashup_api_ds.mashup_num
        self.top_k_list = [3,5,7, 10]
        count = 0
        # assign identifier for api
        for str in mashup_api_ds.api_name:
            self.id_factory['A#' + str] = count  # marked with prefix 'A#'
            count += 1

        # assign identifier for category
        for i in range(len(mashup_api_ds.api_category)):
            for cate in mashup_api_ds.api_category[i]:
                prefix_cat = 'C#' + cate  # marked with prefix 'C#'
                if prefix_cat not in self.id_factory:
                    self.id_factory[prefix_cat] = count
                    count += 1

        for i in range(len(mashup_api_ds.mashup_category)):
            for cate in mashup_api_ds.mashup_category[i]:
                prefix_cat = 'C#' + cate  # marked with prefix 'C#'
                if prefix_cat not in self.id_factory:
                    self.id_factory[prefix_cat] = count
                    count += 1
        # assign identifier for mashup
        for str in mashup_api_ds.mashup_name:
            self.id_factory['M#' + str] = count  # marked with prefix 'M#'
            count += 1
        # create knowledge graph by adding links
        link_source = []
        link_target = []
        link_weight = []

        for idx in range(0,self.num_mashup):
            # adding links between mashups and categories
            new_mashup_idx = self.id_factory['M#' + mashup_api_ds.mashup_name[idx]]
            for cat in mashup_api_ds.mashup_category[idx]:
                new_cat_idx = self.id_factory['C#' + cat]
                link_source.append(new_mashup_idx)
                link_target.append(new_cat_idx)
                link_weight.append(1)

                link_target.append(new_mashup_idx)
                link_source.append(new_cat_idx)
                link_weight.append(1)
        # adding links between mashups and apis
        for m_api in mashup_api_ds.mashup_api:
            mashup = mashup_api_ds.mashup_name[m_api[0]]
            api = mashup_api_ds.api_name[m_api[1]]
            new_api_idx = self.id_factory['A#' + api]
            new_mashup_idx = self.id_factory['M#' + mashup]
            link_source.append(new_mashup_idx)
            link_target.append(new_api_idx)
            link_weight.append(1)

            link_target.append(new_mashup_idx)
            link_source.append(new_api_idx)
            link_weight.append(1)

        # adding links between apis and categories
        for i in range(len(mashup_api_ds.api_category)):
            new_api_idx = self.id_factory['A#' + mashup_api_ds.api_name[i]]
            for cat in mashup_api_ds.api_category[i]:
                new_cat_idx = self.id_factory['C#' + cat]
                link_source.append(new_api_idx)
                link_target.append(new_cat_idx)
                link_weight.append(1)

                link_target.append(new_api_idx)
                link_source.append(new_cat_idx)
                link_weight.append(1)

        self.G = csr_matrix((link_weight, (link_source, link_target)),
                            shape=(len(self.id_factory), len(self.id_factory)))

    def evaluate(self, test_mashup_api_ds):
        print('Start testing ...')
        # API
        ndcg_a = np.zeros(len(self.top_k_list))
        recall_a = np.zeros(len(self.top_k_list))
        ap_a = np.zeros(len(self.top_k_list))
        pre_a = np.zeros(len(self.top_k_list))

        personalize_vector = np.zeros(self.G.shape[0])
        test_idx = test_mashup_api_ds.mashup_api
        print("test_idx",len(test_idx))

        for idx in test_idx:
            mashup_idx = idx[0]
            api_idx = idx[1]
            new_mashup_idx = self.id_factory['M#' + test_mashup_api_ds.mashup_name[mashup_idx]]
            # initialize G and personalize_vector
            personalize_vector[new_mashup_idx] = 1.0
            for cat in test_mashup_api_ds.mashup_category[mashup_idx]:
                new_cat_idx = self.id_factory['C#' + cat]
                self.G[new_mashup_idx, new_cat_idx] = 1

            rwr = pagerank_power(self.G, p=0.85, personalize=personalize_vector, tol=1e-6)
            # recover links between testing mashups and categories
            for cat in test_mashup_api_ds.mashup_category[mashup_idx]:
                new_cat_idx = self.id_factory['C#' + cat]
                self.G[new_mashup_idx, new_cat_idx] = 0
            # recover original G
            self.G.eliminate_zeros()
            personalize_vector[new_mashup_idx] = 0

            # build TOP_N ranking list
            ranklist = sorted(zip(rwr[0:self.num_api], test_mashup_api_ds.api_name), reverse=True)
            # print(ranklist)

            for n in range(len(self.top_k_list)):
                sublist = ranklist[:self.top_k_list[n]]
                score, pred = zip(*sublist)

                l = []
                l.append(test_mashup_api_ds.api_name[api_idx])
                p_at_k = precision(l, pred)
                r_at_k = recall(l, pred)
                ndcg_at_k = ndcg(test_mashup_api_ds.api_name[api_idx], pred)
                ap_at_k = ap(test_mashup_api_ds.api_name[api_idx], pred)

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

        info = '[#Test %d]\n' \
               'NDCG_A:%s\n' \
               'AP_A:%s\n' \
               'Pre_A:%s\n' \
               'Recall_A:%s\n' \
               % (len(test_idx), ndcg_a.round(6), ap_a.round(6), pre_a.round(6), recall_a.round(6))

        print(info)

if __name__ == '__main__':
    # load ds
    print('Start ...')
    train_mashup_api = FCMashupAPI('train_mashup_api.json',is_training=True)
    test_mashup_api = FCMashupAPI('test_mashup_api.json',is_training=False)


    # initial
    rwr = RWR(train_mashup_api)

    rwr.evaluate(test_mashup_api)

# ===============================================================================
