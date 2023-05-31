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
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, SubsetRandomSampler
import os
import sys

from model import util
from model.mashup_api_dataset import MyDataset, MTFMDataset, FCDataset

curPath = os.path.abspath(os.path.dirname('__file__'))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from tools.utils import *
from tools.metric import metric, metric2
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
from datetime import datetime
from tools.metric import recall

def evaluate1(model, test_iter, top_k, device):
    n_processed = 0
    mashup_reprs, api_reprs = [], []
    rec, mrrs, maps, ndcgs = [], [], [], []
    for batch_data in test_iter:
        mashup_desc = batch_data[0].to(device)
        mashup_desc_len = batch_data[1].to(device)
        mashup_category_token = batch_data[2].to(device)
        api_desc = batch_data[3].to(device)
        api_desc_len = batch_data[4].to(device)
        api_category_token = batch_data[5].to(device)

        with torch.no_grad():
            api_pred,mashup_repr,api_repr = model(mashup_desc, mashup_desc_len, mashup_category_token,
                             api_desc, api_desc_len, api_category_token)

        mashup_reprs.append(mashup_repr)
        api_reprs.append(api_repr)
        n_processed += batch_data[0].size(0)
    mashup_reprs = torch.cat(mashup_reprs,dim=0)
    api_reprs = torch.cat(api_reprs,dim=0)

    pool_size = len(mashup_reprs)
    K = 10
    for i in range(pool_size):  # for i in range(pool_size):
        mashup_vec = mashup_reprs[i]  # [1 x dim]
        print(mashup_vec.shape,api_reprs.shape)
        n_results = K
        sims = F.cosine_similarity(api_reprs,mashup_vec)  # [pool_size]
        predict = sims.argsort(descending=True).tolist()
        predict = predict[:n_results]
        #predict = [int(k) for k in predict]
        real = [i]
        rec.append(recall(real, predict))
    print('rec:', np.mean(rec))#rec: 0.019
    return rec

class Config:
    def __init__(self,ds):
        self.model_name = 'FCLSTM'
        self.ds = ds
        self.max_doc_len = self.ds.max_doc_len
        self.max_vocab_size = self.ds.max_vocab_size
        self.embed_dim = self.ds.embed_dim
        self.num_layer = 2
        self.hidden_size = 128
        self.num_category = self.ds.num_category
        self.num_mashup = self.ds.num_mashup
        self.num_api = self.ds.num_api
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.dropout = 0.2
        self.batch_size = 128
        self.lr = 0.05


class FCLSTM(nn.Module):
    def __init__(self, input_config):
        super(FCLSTM, self).__init__()
        self.mashup_embed = nn.Embedding.from_pretrained(input_config.ds.embed, freeze=False)
        self.service_embed = nn.Embedding.from_pretrained(input_config.ds.embed, freeze=False)

        self.mashup_lstm = nn.LSTM(input_size=input_config.embed_dim, hidden_size=input_config.hidden_size,
                                   num_layers=input_config.num_layer, bidirectional=True,
                                   batch_first=True)
        self.service_lstm = nn.LSTM(input_size=input_config.embed_dim, hidden_size=input_config.hidden_size,
                                    num_layers=input_config.num_layer, bidirectional=True,
                                    batch_first=True)

        self.mashup_mlp = nn.Sequential(
            nn.Linear(input_config.embed_dim, input_config.hidden_size*2),
            nn.Sigmoid(),
        )
        self.service_mlp = nn.Sequential(
            nn.Linear(input_config.embed_dim, input_config.hidden_size*2),
            nn.Sigmoid(),
        )

        # self.mashup_fc = nn.Linear(input_config.hidden_size*2, input_config.feature_dim)
        # self.service_fc = nn.Linear(input_config.hidden_size*2, input_config.feature_dim)

        self.tanh = nn.Tanh()
        self.mashup_w = nn.Parameter(torch.zeros(input_config.hidden_size * 2))
        self.service_w = nn.Parameter(torch.zeros(input_config.hidden_size * 2))
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(input_config.num_api*2, input_config.num_api)


    def forward(self, mashup_des, mashup_des_len, mashup_tag, service_des, service_des_len, service_tag):
        mashup_embed = self.mashup_embed(mashup_des)
        packed = pack_padded_sequence(mashup_embed, mashup_des_len.cpu(), batch_first=True, enforce_sorted=False)
        H, _ = self.mashup_lstm(packed)
        H, _ = pad_packed_sequence(H, batch_first=True)
        M = self.tanh(H)
        alpha = F.softmax(torch.matmul(M, self.mashup_w), dim=1).unsqueeze(-1)
        out = H * alpha
        out = torch.sum(out, dim=1)

        mashup_tag_embed = self.mashup_embed(mashup_tag)
        mashup_tag_mlp = self.mashup_mlp(mashup_tag_embed)
        mashup_tag_mlp = mashup_tag_mlp.sum(dim=1).squeeze()

        mashup_att = torch.mul(out, mashup_tag_mlp)
        mashup_feature = torch.cat((mashup_att, mashup_tag_mlp), dim=1)

        service_embed = self.service_embed(service_des)
        packed = pack_padded_sequence(service_embed, service_des_len.cpu(), batch_first=True, enforce_sorted=False)
        H, _ = self.mashup_lstm(packed)
        H, _ = pad_packed_sequence(H, batch_first=True)
        M = self.tanh(H)
        alpha = F.softmax(torch.matmul(M, self.service_w), dim=1).unsqueeze(-1)
        out = H * alpha
        out = torch.sum(out, dim=1)

        service_tag_embed = self.service_embed(service_tag)
        service_tag_mlp = self.service_mlp(service_tag_embed)
        service_tag_mlp = service_tag_mlp.sum(dim=1).squeeze()

        service_att = torch.mul(out, service_tag_mlp)
        service_feature = torch.cat((service_att, service_tag_mlp), dim=1)

        y = F.cosine_similarity(mashup_feature, service_feature)
        return self.sigmoid(y),mashup_feature,service_feature


if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    os.makedirs(f'./runs/{timestamp}_new', exist_ok=True)
    fh = logging.FileHandler(f"./runs/{timestamp}_new/logs.txt")
    logger.addHandler(fh)  # add the handlers to the logger

    # seed for Reproducibility
    util.seed_everything(42)

    # construct the train and test datasets
    ds = FCDataset()
    config = Config(ds=ds)
    train_dataset = ds.mashup_ds
    test_dataset = ds.test_mashup_ds
    train_iter = DataLoader(train_dataset, batch_size=config.batch_size)
    test_iter = DataLoader(test_dataset, batch_size=config.batch_size)

    # set model and loss, optimizer
    model = FCLSTM(input_config=config)
    model = model.to(config.device)
    api_cri = nn.HingeEmbeddingLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr)

    # train, evaluation
    for epoch in range(1, 50 + 1):
        model.train()  # Enable dropout (if have).
        api_loss = []
        for batch_data in train_iter:
            optimizer.zero_grad()
            mashup_desc = batch_data[0].to(config.device)
            mashup_desc_len = batch_data[1].to(config.device)
            mashup_category_token = batch_data[2].to(config.device)

            api_desc = batch_data[3].to(config.device)
            api_desc_len = batch_data[4].to(config.device)
            api_category_token = batch_data[5].to(config.device)
            target = batch_data[6].to(config.device)

            api_pred,_,_ = model(mashup_desc, mashup_desc_len, mashup_category_token,
                             api_desc, api_desc_len, api_category_token)
            api_loss_ = api_cri(api_pred, target)

            api_loss_.backward()
            api_loss.append(api_loss_.item())
            optimizer.step()
        api_loss_ave = np.average(api_loss)
        info = 'ApiLoss:%s' \
               % (api_loss_ave.round(6))
        print(info)
        model.eval()
        # ndcg_a,ap_a,pre_a,recall_a=evaluate(model, test_iter, args.top_k, config.device)
        # ndcg_a, ap_a, pre_a, recall_a = evaluate1(model, test_iter, top_k=[5,10], device=config.device)
        evaluate1(model, test_iter, top_k=[5, 10], device=config.device)
        if epoch % 10 == 0:

            model.eval()
            # ndcg_a,ap_a,pre_a,recall_a=evaluate(model, test_iter, args.top_k, config.device)
            #ndcg_a, ap_a, pre_a, recall_a = evaluate1(model, test_iter, top_k=[5,10], device=config.device)
            evaluate1(model, test_iter, top_k=[5, 10], device=config.device)
            logger.info("The time elapse of epoch {:03d}".format(epoch))
            # logger.info(
            #     "top_5||NDCG: {:.3f}\tAP: {:.3f}\tPRE: {:.3f}\tRecall: {:.3f}".format(ndcg_a[0], ap_a[0], pre_a[0],
            #                                                                           recall_a[0]))
            # logger.info(
            #     "top_10||NDCG: {:.3f}\tAP: {:.3f}\tPRE: {:.3f}\tRecall: {:.3f}".format(ndcg_a[1], ap_a[1], pre_a[1],
            #                                                                            recall_a[1]))
