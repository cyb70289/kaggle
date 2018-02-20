import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class RnnAtt(nn.Module):
    ''' RNN with attention '''

    def __init__(self, embed_dim, hidden_dim, text_len, n_classes, **kwargs):
        ''' kwargs:
            - 'model': 'gru', 'lstm'
            - 'bidir': False, True (bidirectional)
            - 'atten': False, True (add attention)
        '''
        super(RnnAtt, self).__init__()

        self.hidden_dim = hidden_dim
        self.bidir = kwargs.get('bidir', False)
        self.atten = kwargs.get('atten', False)

        model = kwargs.get('model', 'gru').lower()
        if model == 'lstm':
            self.rnn = nn.LSTM(embed_dim, self.hidden_dim, num_layers=1,
                               batch_first=True, bidirectional=self.bidir)
        elif model == 'gru':
            self.rnn = nn.GRU(embed_dim, self.hidden_dim, num_layers=1,
                              batch_first=True, bidirectional=self.bidir)
        else:
            raise(ValueError)

        if self.atten:
            raise(NotImplemented)
        else:
            self.fc = nn.Linear(self.hidden_dim*text_len*(self.bidir+1),
                                n_classes)

    def forward(self, args, text, X):
        # text: batch * sequence * embedding_dim
        h0 = Variable(torch.zeros(1+self.bidir, text.size(0), self.hidden_dim),
                      requires_grad=False)
        if args.cuda:
            h0 = h0.cuda(async=True)
        out, _ = self.rnn(text, h0)
        # flatten: batch * sequence * hidden_dim*(1+bidir) -> batch * -1
        out = out.contiguous()
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out: batch * n_classes
        return out

    def param_options(self):
        return self.parameters()
