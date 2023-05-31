import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tools.utils import *

class JointEmbeder(nn.Module):
    def __init__(self, config,args):
        super(JointEmbeder, self).__init__()
        self.num_mashup=config.num_mashup
        self.num_api = config.num_api
        self.Lstm_encoder = LSTM(config=config)
        self.Ncf_encoder = NeuMF(args=args,num_users=self.num_mashup,num_items=self.num_api)
        self.logistic = nn.Sigmoid()
        self.fc = nn.Linear(self.num_api*2, self.num_api)

    def forward(self, mashup_desc, mashup_desc_len, mashup_index , api_index ):
        # print('code_repr', type(code_repr), code_repr.shape)
        # print('theme_repr', type(theme_repr), theme_repr.shape)
        Lstm_repr = self.Lstm_encoder(mashup_desc, mashup_desc_len)
        Ncf_repr = self.Ncf_encoder(mashup_index , api_index)
        #repr = torch.max(Lstm_repr,Ncf_repr)
        #repr = (Lstm_repr+Ncf_repr)/2
        repr = self.fc(torch.cat([Lstm_repr,Ncf_repr], dim=1))
        return self.logistic(repr)

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
        self.logistic = nn.Sigmoid()

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
        # relu = torch.relu(out)
        fc = self.fc(out)        # category_out = self.fc(out)  # [batch_size, num_category]
        return fc
        #return self.logistic(fc)


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
        # print("vectors:", vector.shape)#vectors: torch.Size([128, 72])
        logits = self.affine_output(vector)
        # print("logits:",logits.shape)
        return logits
        #return self.logistic(logits)