import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

# import config
# import util
# import data_utils
from datetime import datetime
import logging

from model import util
from model.mashup_api_dataset import MyDataset
from tools.metric import metric

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

class NeuMF(nn.Module):
    def __init__(self, args, num_users, num_items):
        super(NeuMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.factor_num_mf = args.factor_num
        self.factor_num_mlp = int(args.layers[0] / 2)
        self.layers = args.layers
        self.dropout = args.dropout

        self.embedding_user_mlp = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num_mlp)
        self.embedding_item_mlp = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num_mlp)

        self.embedding_user_mf = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num_mf)
        self.embedding_item_mf = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num_mf)

        self.fc_layers = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(args.layers[:-1], args.layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.fc_layers.append(nn.ReLU())

        self.affine_output = nn.Linear(in_features=args.layers[-1] + self.factor_num_mf, out_features=num_items)
        self.logistic = nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.embedding_user_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_user_mf.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mf.weight, std=0.01)

        for m in self.fc_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        nn.init.xavier_uniform_(self.affine_output.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)

        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

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
            mashup_desc_len = batch_data[1].to(device)
            mashup_index = batch_data[4].to(device)
            api_index = batch_data[5].to(device)
            api_target = batch_data[6].to(device)
            api_pred = model(mashup_index, api_index)

            api_loss_ = loss_function(api_pred, api_target)
            api_loss.append(api_loss_.item())
            api_pred = api_pred.cpu().detach()
            #ndcg_, recall_, ap_, pre_ = metric3(batch_data[6], api_pred, top_k)
            ndcg_, recall_, ap_, pre_ = metric(batch_data[6], api_pred, top_k)
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Seed")
    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="learning rate")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.2,
                        help="dropout rate")
    parser.add_argument("--batch_size",
                        type=int,
                        default=128,
                        help="batch size for training")
    parser.add_argument("--epochs",
                        type=int,
                        default=50,
                        help="training epoches")
    parser.add_argument("--top_k",
                        type=int,
                        default=10,
                        help="compute metrics@top_k")
    parser.add_argument("--top_k_list",
                        type=int,
                        default=[3,5,7, 10],
                        help="compute metrics@top_k")
    parser.add_argument("--factor_num",
                        type=int,
                        default=64,
                        help="predictive factors numbers in the model")
    parser.add_argument("--layers",
                        nargs='+',
                        default=[64, 32, 16, 8],
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument("--num_ng",
                        type=int,
                        default=4,
                        help="Number of negative samples for training set")
    parser.add_argument("--num_ng_test",
                        type=int,
                        default=10,
                        help="Number of negative samples for test set")
    parser.add_argument("--out",
                        default=True,
                        help="save model or not")

    return parser.parse_args()

if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    os.makedirs(f'./runs/{timestamp}_new', exist_ok=True)
    fh = logging.FileHandler(f"./runs/{timestamp}_new/logs.txt")
    logger.addHandler(fh)  # add the handlers to the logger

    # set device and parameters
    args = parse_args()
    writer = SummaryWriter(f'./runs/{timestamp}_new')
    ds = MyDataset()
    # seed for Reproducibility
    util.seed_everything(args.seed)


    # construct the train and test datasets

    train_dataset = ds.mashup_ds
    test_dataset = ds.test_mashup_ds
    train_iter = DataLoader(train_dataset, batch_size=args.batch_size)
    test_iter = DataLoader(test_dataset, batch_size=args.batch_size)

    # set model and loss, optimizer
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeuMF(args,ds.num_mashup,ds.num_api)
    model = model.to(device)
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # train, evaluation
    best_hr = 0
    for epoch in range(1, args.epochs + 1):
        model.train()  # Enable dropout (if have).
        api_loss = []
        for batch_data in train_iter:
            optimizer.zero_grad()
            mashup_desc = batch_data[0].to(device)
            mashup_desc_len = batch_data[1].to(device)
            api_desc = batch_data[2].to(device)
            api_desc_len = batch_data[3].to(device)
            mashup_index = batch_data[4].to(device)
            api_index = batch_data[5].to(device)
            api_target = batch_data[6].to(device)
            api_pred = model( mashup_index , api_index)
            api_loss_ = loss_function(api_pred, api_target)
            api_loss_.backward()
            api_loss.append(api_loss_.item())
            optimizer.step()

        if epoch % 10 == 0:
            api_loss = np.average(api_loss)
            info = 'ApiLoss:%s\n' \
                   % (api_loss.round(6))
            print(info)
            model.eval()
            ndcg_a,ap_a,pre_a,recall_a=evaluate(model, test_iter, args.top_k_list, device)
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

    writer.close()
#NCF
#top_5||NDCG: 0.877	AP: 0.875	PRE: 0.177	Recall: 0.883
#top_10||NDCG: 0.878	AP: 0.875	PRE: 0.089	Recall: 0.887

#top_5||NDCG: 0.884	AP: 0.882	PRE: 0.178	Recall: 0.890
#top_10||NDCG: 0.885	AP: 0.882	PRE: 0.089	Recall: 0.892