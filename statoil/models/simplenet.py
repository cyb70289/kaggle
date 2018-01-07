import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvLayer(nn.Sequential):

    def __init__(self, conv_lst, n_channels, dropout):
        super(_ConvLayer, self).__init__()

        # conv layers
        for i, channels_lst in enumerate(conv_lst, 1):
            # conv + relu + pool
            if not isinstance(channels_lst, (list, tuple)):
                channels_lst = [channels_lst]
            add_bn = len(channels_lst) > 1
            for j, channels in enumerate(channels_lst, 1):
                self.add_module('conv{}-{}'.format(i, j),
                                nn.Conv2d(n_channels, channels, 3, padding=1,
                                          bias=False))
                if add_bn:
                    self.add_module('norm{}-{}'.format(i, j),
                                    nn.BatchNorm2d(channels))
                self.add_module('relu{}-{}'.format(i, j),
                                nn.ReLU(inplace=True))
                n_channels = channels
            # skip last layer
            if i < len(conv_lst):
                self.add_module('pool{}'.format(i), nn.MaxPool2d(2, 2))
                if dropout > 0.001:
                    self.add_module('dropout{}'.format(i), nn.Dropout(dropout))

        self.n_channels = n_channels


class _FCLayer(nn.Sequential):

    def __init__(self, channels_lst, n_channels, n_classes, dropout):
        super(_FCLayer, self).__init__()

        if not isinstance(channels_lst, (list, tuple)):
            channels_lst = [channels_lst]
        for i, channels in enumerate(channels_lst, 1):
            self.add_module('linear{}'.format(i),
                            nn.Linear(n_channels, channels))
            self.add_module('relu{}'.format(i), nn.ReLU(inplace=True))
            if dropout > 0.001:
                self.add_module('dropout{}'.format(i), nn.Dropout(dropout))
            n_channels = channels

        self.add_module('linear{}'.format(i+1),
                        nn.Linear(n_channels, n_classes))


class SimpleNet(nn.Module):

    _def_layer_dict = {
        'conv': (64, 128, 256, 512),
        'fc': (1024, 1024),
    }

    def __init__(self, layer_dict=_def_layer_dict, n_channels=2,
                 n_classes=1, dropout_conv=0.5, dropout_fc=0.5):
        super(SimpleNet, self).__init__()

        self.conv_layer = _ConvLayer(layer_dict['conv'], n_channels,
                                     dropout_conv)
        self.fc_layer = _FCLayer(layer_dict['fc'], self.conv_layer.n_channels,
                                 n_classes, dropout_fc)

    def forward(self, *X):
        X_img = X[0]
        # conv
        y = self.conv_layer(X_img)
        # global pool
        poolsize = y.size()[-2:]
        y = F.avg_pool2d(y, poolsize)
        y = y.view(y.size(0), -1)
        y = F.relu(y)
        # fc
        return self.fc_layer(y)

    def param_options(self):
        # no special options
        return self.parameters()
