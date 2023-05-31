import os
import argparse
import numpy as np

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from datetime import datetime
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model import util
from model.mashup_api_dataset import MyDataset
from tools.metric import metric

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


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
            api_pred = model(mashup_desc, mashup_desc_len)

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
                        default=300,
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

class LSTMConfig(object):
    def __init__(self):
        super(LSTMConfig, self).__init__()
        self.ds = MyDataset()
        self.model_name = 'LSTM'
        self.embed_dim = self.ds.embed_dim
        self.hidden_size = 128
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 128
        self.max_doc_len = self.ds.max_doc_len
        self.top_k_list = [1, 5, 10, 15, 20, 25, 30]
        self.lr = 0.001
        self.dropout = 0.2
        self.num_layer = 1

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
        packed = pack_padded_sequence(sorted_embed, sorted_lengths.cpu(), batch_first=True)
        H, _ = self.lstm(packed)  # [batch_size, seq_len, hidden_size * num_direction]
        H, _ = pad_packed_sequence(H, batch_first=True)

        desorted_H = H[desorted_indices]
        M = self.tanh(desorted_H)  # [batch_size, seq_len, hidden_size * num_direction]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [batch_size, seq_len, 1]
        out = H * alpha  # [batch_size, seq_len, hidden_size * num_direction]
        out = torch.sum(out, dim=1)  # [batch_size, hidden_size * num_direction]
        #print("out",out.shape)
        # relu = torch.relu(out)
        fc = self.fc(out)        # category_out = self.fc(out)  # [batch_size, num_category]
        return torch.sigmoid(fc)

if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    os.makedirs(f'./runs/{timestamp}_new', exist_ok=True)
    fh = logging.FileHandler(f"./runs/{timestamp}_new/logs.txt")
    logger.addHandler(fh)  # add the handlers to the logger

    # set device and parameters
    args = parse_args()
    writer = SummaryWriter(f'./runs/{timestamp}_new')
    config = LSTMConfig()
    ds = config.ds
    # seed for Reproducibility
    util.seed_everything(args.seed)


    # construct the train and test datasets

    train_dataset = ds.mashup_ds
    test_dataset = ds.test_mashup_ds
    train_iter = DataLoader(train_dataset, batch_size=args.batch_size)
    test_iter = DataLoader(test_dataset, batch_size=args.batch_size)

    # set model and loss, optimizer
    model = LSTM(config)
    model = model.to(config.device)
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # train, evaluation
    best_hr = 0
    for epoch in range(1, 50 + 1):
        model.train()  # Enable dropout (if have).
        api_loss = []
        for batch_data in train_iter:
            optimizer.zero_grad()
            mashup_desc = batch_data[0].to(config.device)
            mashup_desc_len = batch_data[1].to(config.device)
            api_desc = batch_data[2].to(config.device)
            api_desc_len = batch_data[3].to(config.device)
            mashup_index = batch_data[4].to(config.device)
            api_index = batch_data[5].to(config.device)
            api_target = batch_data[6].to(config.device)
            api_pred = model(mashup_desc, mashup_desc_len)
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
            ndcg_a,ap_a,pre_a,recall_a=evaluate(model, test_iter, args.top_k_list, config.device)
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

