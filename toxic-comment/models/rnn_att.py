import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class RnnAtt(nn.Module):
    ''' RNN with attention '''

    def __init__(self, input_dim, hidden_dim, seqlen, n_classes, **kwargs):
        ''' kwargs:
            - 'model': 'gru', 'lstm'
            - 'bidir': 0, 1 (bidirectional)
            - 'atten': 0, 1 (add attention)
        '''
        super(RnnAtt, self).__init__()

        self.hidden_dim = hidden_dim
        self.bidir = kwargs.get('bidir', 0)
        self.atten = kwargs.get('atten', 0)
        model = kwargs.get('model', 'gru').lower()
        assert(self.bidir == 0 or self.bidir == 1)
        assert(self.atten == 0 or self.atten == 1)

        if model == 'lstm':
            self.rnn = nn.LSTM(input_dim, self.hidden_dim, num_layers=1,
                               batch_first=True, bidirectional=self.bidir)
        elif model == 'gru':
            self.rnn = nn.GRU(input_dim, self.hidden_dim, num_layers=1,
                              batch_first=True, bidirectional=self.bidir)
        else:
            raise(ValueError)

        if self.atten:
            raise(NotImplemented)
        else:
            self.fc = nn.Linear(self.hidden_dim*seqlen*(self.bidir+1),
                                n_classes)

    def forward(self, text, X):
        # text: batch * sequence * embedding_dim
        hidden = Variable(
            torch.zeros(1+self.bidir, text.size(0), self.hidden_dim),
            requires_grad=False)
        out, _ = self.rnn(text, hidden)
        # flatten: batch * sequence * hidden_dim*(1+bidir) -> batch * -1
        out = out.view(text.size(0), -1)
        out = self.fc(out)
        # out: batch * n_classes
        return out

    def param_options(self):
        return self.parameters()
