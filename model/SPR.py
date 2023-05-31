# -*- conding: utf-8 -*-
"""
@File   : MTFM.py
@Time   : 2021/3/15
@Author : yhduan
@Desc   : None
"""
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import os
import sys

from model import util
from model.mashup_api_dataset import MyDataset, MTFMDataset, M_Dataset

curPath = os.path.abspath(os.path.dirname('__file__'))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from tools.dataset_class import *
from tools.utils import *
from tools.metric import metric, metric2
import logging
from gensim import corpora, similarities
from gensim.models import AuthorTopicModel
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
from datetime import datetime

def evaluate(model, test_iter, top_k, device):
    ndcg_a = 0
    recall_a = 0
    ap_a = 0
    pre_a = 0
    api_loss = []
    loss_function = nn.BCELoss()
    num_batch = len(test_iter)
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_iter):

            api_target = batch_data[1].to(config.device)
            api_pred = model(batch_data[4].to(config.device))

            api_loss_ = loss_function(api_pred, api_target)
            api_loss.append(api_loss_.item())
            api_pred = api_pred.cpu().detach()
            #ndcg_, recall_, ap_, pre_ = metric3(batch_data[6], api_pred, top_k)
            ndcg_, recall_, ap_, pre_ = metric(batch_data[1], api_pred, top_k)
            ndcg_a += ndcg_
            recall_a += recall_
            ap_a += ap_
            pre_a += pre_
    api_loss = np.average(api_loss)
    ndcg_a /= num_batch
    recall_a /= num_batch
    ap_a /= num_batch
    pre_a /= num_batch
    info = 'ApiLoss:%s\n' \
           'NDCG_A:%s\n' \
           'AP_A:%s\n' \
           'Pre_A:%s\n' \
           'Recall_A:%s\n' % ( api_loss, ndcg_a,
                              ap_a, pre_a, recall_a)
    return ndcg_a,ap_a,pre_a,recall_a

class SPRConfig(object):
    def __init__(self):
        super(SPRConfig, self).__init__()
        self.mashup_ds = M_Dataset('train_mashup_api.json')
        self.model_name = 'SPR'
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device = ('cpu')
        self.batch_size = 128
        self.num_mashup = self.mashup_ds.mashup_num
        self.num_api = self.mashup_ds.api_num
        self.feature_size = 8
        self.lr = 1e-3
        self.h = 1  # Thresholds for dominant words generation
        self.g = 0.5  # Thresholds for dominant words generation
        self.dictionary = corpora.Dictionary(self.mashup_ds.mashup_description)
        author2doc = {}
        self.doc_term_matrix = [self.dictionary.doc2bow(doc) for doc in self.mashup_ds.mashup_description]
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
        self.RSP = np.zeros((self.num_api, len(self.dictionary)))
        for i, docid in enumerate(self.ATM.author2doc.values()):
            for doc in docid:
                self.RSP[i] += self.M2D[doc]*0.2
        self.RSP += 0.5
        self.w_sum = self.RSP.sum(axis=0)
        self.P = self.RSP/self.w_sum
        self.DW = np.zeros((self.num_api, len(self.dictionary)))
        for s in range(self.num_api):
            for w in range(len(self.dictionary)):
                if self.RSP[s, w] >= self.h and self.P[s, w] >= self.g:
                    self.DW[s, w] = 0.1

        self.L = []
        for des in self.mashup_ds.mashup_description:
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
        #print('self.m2d:',self.m2d.shape)
        self.a2d = torch.tensor(self.P)
        self.R = F.normalize(torch.tensor(self.R))

class SPR(torch.nn.Module):
    def __init__(self, config):
        super(SPR, self).__init__()
        self.m2d_mat = nn.Embedding.from_pretrained(config.m2d.float(), freeze=False)
        self.a2d_mat = nn.Embedding.from_pretrained(config.a2d.float(), freeze=False)
        self.R = nn.Embedding.from_pretrained(config.R.float(), freeze=False)
        self.logitic = nn.Sigmoid()

    def forward(self, user_indices):
        m2d = self.m2d_mat(user_indices)
        a2d = self.a2d_mat.weight.t()
        r = F.normalize(torch.mm(m2d, a2d))
        r += self.R(user_indices)
        return self.logitic(r)

if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    os.makedirs(f'./runs/{timestamp}_new', exist_ok=True)
    fh = logging.FileHandler(f"./runs/{timestamp}_new/logs.txt")
    logger.addHandler(fh)  # add the handlers to the logger

    ds = MTFMDataset()
    # set device and parameters
    config = SPRConfig()
    # seed for Reproducibility
    util.seed_everything(42)

    # construct the train and test datasets

    train_dataset = ds.mashup_ds
    test_dataset = ds.test_mashup_ds
    train_iter = DataLoader(train_dataset, batch_size=config.batch_size)
    test_iter = DataLoader(test_dataset, batch_size=config.batch_size)

    # set model and loss, optimizer
    model = SPR(config=config)
    model = model.to(config.device)
    loss_function = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # train, evaluation
    for epoch in range(1, 50 + 1):
        model.train()  # Enable dropout (if have).
        api_loss = []
        for batch_data in train_iter:
            optimizer.zero_grad()
            api_target = batch_data[1].to(config.device)

            api_pred = model(batch_data[4].to(config.device))
            api_loss_ = loss_function(api_pred, api_target)
            api_loss_.backward()
            api_loss.append(api_loss_.item())
            optimizer.step()
        api_loss_ave = np.average(api_loss)
        info = 'ApiLoss:%s' \
               % (api_loss_ave.round(6))
        print(info)
        if epoch % 10 == 0:
            model.eval()
            # ndcg_a,ap_a,pre_a,recall_a=evaluate(model, test_iter, args.top_k, config.device)
            ndcg_a, ap_a, pre_a, recall_a = evaluate(model, test_iter, top_k=[3,5,7,10], device=config.device)
            logger.info("The time elapse of epoch {:03d}".format(epoch))
            logger.info(
                "top_3||NDCG: {:.4f}\tAP: {:.4f}\tPRE: {:.3f}\tRecall: {:.4f}".format(ndcg_a[0], ap_a[0], pre_a[0],
                                                                                      recall_a[0]))
            logger.info(
                "top_5||NDCG: {:.4f}\tAP: {:.4f}\tPRE: {:.3f}\tRecall: {:.4f}".format(ndcg_a[1], ap_a[1], pre_a[1],
                                                                                      recall_a[1]))
            logger.info(
                "top_7||NDCG: {:.4f}\tAP: {:.4f}\tPRE: {:.3f}\tRecall: {:.4f}".format(ndcg_a[2], ap_a[2], pre_a[2],
                                                                                      recall_a[2]))
            logger.info(
                "top_10||NDCG: {:.4f}\tAP: {:.4f}\tPRE: {:.3f}\tRecall: {:.4f}".format(ndcg_a[3], ap_a[3], pre_a[3],
                                                                                       recall_a[3]))
