import torch
import torch.nn as nn
import torch.nn.functional as F


class _InputLayer(nn.Sequential):

    def __init__(self, densenet, n_channels):
        super(_InputLayer, self).__init__()

        # do nothing
        self.n_channels = n_channels


class _DenseLayer(nn.Sequential):

    def __init__(self, densenet, n_channels):
        super(_DenseLayer, self).__init__()
        grow_rate = densenet.grow_rate

        # output channels
        self.n_channels = grow_rate + n_channels

        if densenet.add_bottleneck:
            self.add_module('norm1', nn.BatchNorm2d(n_channels))
            self.add_module('relu1', nn.ReLU(inplace=True))
            self.add_module('conv1', nn.Conv2d(n_channels, 4*grow_rate, 1,
                                               bias=False))
            n_channels = 4*grow_rate

        self.add_module('norm2', nn.BatchNorm2d(n_channels))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(n_channels, grow_rate, 3, padding=1,
                                           bias=False))

    def forward(self, x):
        y = super(_DenseLayer, self).forward(x)
        return torch.cat([x, y], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, densenet, n_layers, n_channels):
        super(_DenseBlock, self).__init__()

        for i in range(n_layers):
            layer = _DenseLayer(densenet, n_channels)
            n_channels = layer.n_channels
            self.add_module('denselayer{}'.format(i+1), layer)

        self.n_channels = n_channels


class _TransLayer(nn.Sequential):

    def __init__(self, densenet, n_channels, is_last):
        super(_TransLayer, self).__init__()
        self.is_last = is_last
        self.n_channels = n_channels

        self.add_module('norm', nn.BatchNorm2d(n_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        if not is_last:
            self.n_channels = int(n_channels * densenet.compress_rate)
            self.add_module('conv', nn.Conv2d(n_channels, self.n_channels, 1,
                                              bias=False))
            if densenet.dropout_rate > 0.0001:
                self.add_module('dropout', nn.Dropout(densenet.dropout_rate,
                                                      inplace=True))
            self.add_module('pool', nn.AvgPool2d(2, 2))


class _OutputLayer(nn.Module):

    def __init__(self, densenet, n_channels):
        super(_OutputLayer, self).__init__()
        self.nn_linear = nn.Linear(n_channels, densenet.n_classes)

    def forward(self, x):
        # global pooling
        poolsize = x.size()[-2:]
        y = F.avg_pool2d(x, poolsize)
        # flatten
        y = y.view(y.size(0), -1)
        # linear
        y = self.nn_linear(y)
        return y


class DenseNet(nn.Sequential):

    def __init__(self, block_lst=(6, 12, 24, 16), n_channels=2,
                 n_classes=1, grow_rate=12, compress_rate=0.5,
                 add_bottleneck=True, dropout_rate=0.0):
        super(DenseNet, self).__init__()
        if n_classes == 2:
            n_classes = 1
        n_blocks = len(block_lst)
        self.n_classes = n_classes
        self.grow_rate = grow_rate
        self.compress_rate = compress_rate
        self.add_bottleneck = add_bottleneck
        self.dropout_rate = dropout_rate

        layer = _InputLayer(self, n_channels)
        n_channels = layer.n_channels
        self.add_module('Input', layer)

        for i in range(n_blocks):
            layer = _DenseBlock(self, block_lst[i], n_channels)
            self.add_module('DenseBlock{}'.format(i+1), layer)
            n_channels = layer.n_channels

            layer = _TransLayer(self, n_channels, i==n_blocks-1)
            self.add_module('Transition{}'.format(i+1), layer)
            n_channels = layer.n_channels

        self.add_module('Output', _OutputLayer(self, n_channels))

    def forward(self, *X):
        X_img = X[0]
        return super(DenseNet, self).forward(X_img)

    def param_options(self):
        # don't regularize batchnorm layers
        params_batchnorm = []
        params_others = []
        for name, param in dict(self.named_parameters()).items():
            if '.norm' in name:
                params_batchnorm.append(param)
            else:
                params_others.append(param)
        options = [
            { 'params': params_others },
            { 'params': params_batchnorm, 'weight_decay': 0 },
        ]
        return options
