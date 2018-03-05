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
        query_dim = kwargs.get('query_dim', 32)
        fc_dim = kwargs.get('fc_dim', 1024)

        self.query_weights = nn.Parameter(torch.zeros(embed_dim, query_dim))
        if cuda:
            self.query_weights.cuda(async=True)
        self.fc1 = nn.Linear(query_dim*embed_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, n_classes)

    def forward(self, text, X):
        # text: batch * sequence * embed_dim
        atten = torch.matmul(text, self.query_weights)
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
        # out: batch * fc_dim
        out = self.fc2(out)
        # out: batch * n_classes
        return out

    def param_options(self):
        return self.parameters()
