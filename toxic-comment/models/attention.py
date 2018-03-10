import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SimpleAtt(nn.Module):

    def __init__(self, embed_dim, text_len, n_classes, **kwargs):
        super(SimpleAtt, self).__init__()
        self._cuda = kwargs.get('cuda', False)
        self.query_weights = nn.Parameter(torch.zeros(embed_dim))
        if self._cuda:
            self.query_weights.cuda(async=True)
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, text, X):
        # text: batch * sequence * embed_dim
        atten_weights = (text * self.query_weights).sum(dim=-1)
        atten_weights = F.softmax(atten_weights, dim=1)
        # atten_weights: batch * sequence
        out = text * atten_weights[..., None]
        out = out.sum(dim=1)
        # out: batch * embed_dim
        out = self.fc(out)
        # out: batch * n_classes
        return out

    def param_options(self):
        return self.parameters()


class SelfAtt(nn.Module):

    def __init__(self, embed_dim, text_len, n_classes, **kwargs):
        super(SelfAtt, self).__init__()
        cuda = kwargs.get('cuda', False)
        query_dim = kwargs.get('query_dim', 6)
        fc_dim = kwargs.get('fc_dim', 1024)

        self.pe = self.get_position_encoding(text_len, embed_dim)
        self.query_weights = nn.Parameter(torch.zeros(embed_dim, query_dim))
        self.fc1 = nn.Linear(query_dim*embed_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, n_classes)
        if cuda:
            self.query_weight = self.query_weights.cuda(async=True)
            self.pe = self.pe.cuda(async=True)

    def get_position_encoding(self, text_len, embed_dim):
        pe = np.empty((text_len, embed_dim), dtype=np.float32)
        ieven = np.arange(0, embed_dim, 2)
        iodd = np.arange(1, embed_dim, 2)
        for pos in range(text_len):
            pe[pos, ieven] = np.sin(pos / np.power(10000, ieven/embed_dim))
            pe[pos, iodd] = np.cos(pos / np.power(10000, (iodd-1)/embed_dim))
        pe = pe[None, ...]
        return Variable(torch.from_numpy(pe), requires_grad=False)

    def forward(self, text, X):
        # text: batch * sequence * embed_dim
        atten = text + self.pe
        atten = torch.matmul(atten, self.query_weights)
        atten = F.softmax(atten, dim=1)
        # atten: batch * sequence * query_dim
        atten = atten.transpose(1, 2)
        # atten: batch * query_dim * sequence
        atten = torch.matmul(atten, text)
        # atten: batch * query_dim * embed_dim
        atten = atten.view(atten.size(0), -1)
        # atten: batch * (query_dim*embed_dim)
        out = self.fc1(atten)
        out = F.relu(out, inplace=True)
        # out = F.dropout(out, 0.1, training=True)
        # out: batch * fc_dim
        out = self.fc2(out)
        # out: batch * n_classes
        return out

    def param_options(self):
        return self.parameters()

    def reg_loss(self):
        # normalize column vectors
        l2sum = (self.query_weights ** 2).sum(dim=0, keepdim=True)
        weights = self.query_weights / (torch.sqrt(l2sum + 1e-16))
        # f2_mtx[i, j] = col_i .dot. col_j
        f2_mtx = torch.mm(weights.transpose(0, 1), weights)
        # clear trace
        idx = np.arange(f2_mtx.size(0))
        f2_mtx[idx, idx] = 0
        # return regularization loss
        loss = (f2_mtx ** 2).mean()
        return 0.001 * loss
