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
from model.mashup_api_dataset import MyDataset, MTFMDataset

curPath = os.path.abspath(os.path.dirname('__file__'))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from tools.dataset_class import *
from tools.utils import *
from tools.metric import metric, metric2
import logging
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
            mashup_desc = batch_data[0].to(device)

            api_target = batch_data[1].to(device)
            api_pred, _ = model(mashup_desc)

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

class MTFMConfig(object):
    def __init__(self, ds_config):
        self.model_name = 'MTFM'
        self.embed_dim = ds_config.embed_dim
        self.max_doc_len = ds_config.max_doc_len
        self.dropout = 0.2
        self.num_category = ds_config.num_category
        self.feature_dim = 8
        self.num_kernel = 256
        self.dropout = 0.2
        self.kernel_size = [2, 3, 4, 5]
        self.num_mashup = ds_config.num_mashup
        self.num_api = ds_config.num_api
        self.vocab_size = ds_config.vocab_size
        self.embed = ds_config.embed
        self.lr = 1e-4
        self.batch_size = 128
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')

class MTFM(nn.Module):
    def __init__(self, config):
        super(MTFM, self).__init__()
        if config.embed is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embed, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.vocab_size - 1)

        self.sc_convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=config.embed_dim,
                                    out_channels=config.num_kernel,
                                    kernel_size=h),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=config.max_doc_len-h+1))
            for h in config.kernel_size
        ])
        self.sc_fcl = nn.Linear(in_features=config.num_kernel * len(config.kernel_size),
                                out_features=config.num_api)

        self.fic_fc = nn.Linear(in_features=config.num_kernel * len(config.kernel_size),
                                out_features=config.feature_dim)
        self.fic_api_feature_embedding = nn.Parameter(torch.rand(config.feature_dim, config.num_api))
        self.fic_mlp = nn.Sequential(
            nn.Linear(config.feature_dim*2, config.feature_dim),
            nn.Linear(config.feature_dim, 1),
            nn.Tanh()
        )
        self.fic_fcl = nn.Linear(config.num_api*2, config.num_api)

        self.fusion_layer = nn.Linear(config.num_api*2, config.num_api)

        self.api_task_layer = nn.Linear(config.num_api, config.num_api)
        self.category_task_layer = nn.Linear(config.num_api, config.num_category)

        self.dropout = nn.Dropout(config.dropout)
        self.logistic = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def init_weight(self):
        nn.init.kaiming_normal_(self.fic_api_feature_embedding)

    def forward(self, mashup_des):
        # semantic component
        embed = self.embedding(mashup_des)
        embed = embed.permute(0, 2, 1)
        e = [conv(embed) for conv in self.sc_convs]
        e = torch.cat(e, dim=2)
        e = e.view(e.size(0), -1)
        u_sc = self.sc_fcl(e)
        # u_sc = self.tanh(u_sc)

        # feature interaction component
        u_sc_trans = self.fic_fc(e)
        u_mm = torch.matmul(u_sc_trans, self.fic_api_feature_embedding)
        u_concate = []
        for u_sc_single in u_sc_trans:
            u_concate_single = torch.cat((u_sc_single.repeat(self.fic_api_feature_embedding.size(1), 1), self.fic_api_feature_embedding.t()), dim=1)
            u_concate.append(self.fic_mlp(u_concate_single).squeeze())
        u_mlp = torch.cat(u_concate).view(u_mm.size(0), -1)
        u_fic = self.fic_fcl(torch.cat((u_mm, u_mlp), dim=1))
        u_fic = self.tanh(u_fic)

        # fusion layer
        u_mmf = self.fusion_layer(torch.cat((u_sc, u_fic), dim=1))

        # dropout
        u_mmf = self.dropout(u_mmf)

        # api-specific task layer
        y_m = self.api_task_layer(u_mmf)

        # mashup category-specific task layer
        z_m = self.category_task_layer(u_mmf)

        return self.logistic(y_m), self.logistic(z_m)

if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    os.makedirs(f'./runs/{timestamp}_new', exist_ok=True)
    fh = logging.FileHandler(f"./runs/{timestamp}_new/logs.txt")
    logger.addHandler(fh)  # add the handlers to the logger

    ds = MTFMDataset()
    # set device and parameters
    config = MTFMConfig(ds)
    # seed for Reproducibility
    util.seed_everything(42)

    # construct the train and test datasets

    train_dataset = ds.mashup_ds
    test_dataset = ds.test_mashup_ds
    train_iter = DataLoader(train_dataset, batch_size=config.batch_size)
    test_iter = DataLoader(test_dataset, batch_size=config.batch_size)

    # set model and loss, optimizer
    model = MTFM(config=config)
    model = model.to(config.device)
    api_cri = torch.nn.BCELoss()
    cate_cri = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # train, evaluation
    for epoch in range(1, 100 + 1):
        model.train()  # Enable dropout (if have).
        api_loss = []
        for batch_data in train_iter:
            optimizer.zero_grad()
            mashup_desc = batch_data[0].to(config.device)
            api_target = batch_data[1].to(config.device)
            category_target = batch_data[2].float().to(config.device)

            api_pred , category_pred = model(mashup_desc)
            api_loss_ = api_cri(api_pred, api_target)
            category_loss_ = cate_cri(category_pred, category_target)

            loss_ = category_loss_ + api_loss_
            loss_.backward()
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
