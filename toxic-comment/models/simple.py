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
        # text: batch * sequence * embedding_dim
        atten_weights = (text * self.query_weights).sum(dim=-1)
        atten_weights = F.softmax(atten_weights, dim=1)
        # atten_weights: batch * sequence
        out = text * atten_weights[..., None]
        out = out.sum(dim=1)
        # out: batch * embedding_dim
        out = self.fc(out)
        # out: batch * n_classes
        return out

    def param_options(self):
        return self.parameters()
