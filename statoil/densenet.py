import torch
import torch.nn as nn
import torch.nn.functional as F


class _InputLayer(nn.Module):
    def __init__(self, densenet, n_channels):
        super(_InputLayer, self).__init__()

        if densenet.add_bottleneck and densenet.compress_rate < 1:
            self.n_channels = densenet.grow_rate * 2
        else:
            self.n_channels = 16
        self.nn_conv = nn.Conv2d(n_channels, self.n_channels, 3, padding=1,
                                 bias=False)

    def forward(self, x):
        return self.nn_conv(x)


class _DenseLayer1(nn.Module):
    def __init__(self, densenet, n_channels):
        super(_DenseLayer1, self).__init__()
        self.densenet = densenet
        grow_rate = densenet.grow_rate

        # output channels
        self.n_channels = grow_rate + n_channels

        if densenet.add_bottleneck:
            self.nn_bn_bottle = nn.BatchNorm2d(n_channels)
            self.nn_conv_bottle = nn.Conv2d(n_channels, 4*grow_rate, 1,
                                            bias=False)
            n_channels = 4*grow_rate

        self.nn_bn = nn.BatchNorm2d(n_channels)
        self.nn_conv = nn.Conv2d(n_channels, grow_rate, 3, padding=1,
                                 bias=False)

    def forward(self, x):
        y = x

        if self.densenet.add_bottleneck:
            y = self.nn_bn_bottle(y)
            y = F.relu(y)
            y = self.nn_conv_bottle(y)

        y = self.nn_bn(y)
        y = F.relu(y)
        y = self.nn_conv(y)

        return torch.cat([y, x], 1)


class _DenseLayer(nn.Module):
    def __init__(self, densenet, n_layers, n_channels):
        super(_DenseLayer, self).__init__()
        self.n_layers = n_layers
        self.layer_lst = []

        for i in range(n_layers):
            layer = _DenseLayer1(densenet, n_channels)
            self.layer_lst.append(layer)
            n_channels = layer.n_channels

        self.n_channels = n_channels

    def forward(self, x):
        y = x
        for i in range(self.n_layers):
            y = self.layer_lst[i](y)
        return y


class _TransLayer(nn.Module):
    def __init__(self, densenet, n_channels, is_last):
        super(_TransLayer, self).__init__()
        self.is_last = is_last

        self.n_channels = n_channels
        self.nn_bn = nn.BatchNorm2d(n_channels)
        if not is_last:
            out_channels = int(n_channels * densenet.compress_rate)
            self.nn_conv = nn.Conv2d(n_channels, out_channels, 1, bias=False)
            self.n_channels = out_channels

    def forward(self, x):
        y = self.nn_bn(x)
        y = F.relu(y)
        if not self.is_last:
            y = self.nn_conv(y)
            y = F.avg_pool2d(y, 2)
        return y


class _OutputLayer(nn.Module):
    def __init__(self, densenet, n_channels):
        super(_OutputLayer, self).__init__()

        self.nn_linear = nn.Linear(n_channels, densenet.n_classes)
        self.nn_sigmoid = nn.Sigmoid()

    def forward(self, x):
        # global pooling
        poolsize = x.size()[-2:]
        y = F.avg_pool2d(x, poolsize)
        # flatten
        y = y.view(y.size()[0], -1)
        # linear
        y = self.nn_linear(y)
        y = self.nn_sigmoid(y)
        return y


class DenseNet(nn.Module):
    def __init__(self, block_lst, n_channels=2, n_classes=1, grow_rate=12,
                 compress_rate=0.5, add_bottleneck=True, dropout_rate=0.0):
        super(DenseNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.grow_rate = grow_rate
        self.compress_rate = compress_rate
        self.add_bottleneck = add_bottleneck
        self.dropout_rate = dropout_rate
        self.block_lst = block_lst
        self.n_blocks = len(block_lst)

        self.input_layer = _InputLayer(self, n_channels)
        n_channels = self.input_layer.n_channels

        self.dense_layer, self.trans_layer = [], []
        for i in range(self.n_blocks):
            # dense layer
            layer = _DenseLayer(self, block_lst[i], n_channels)
            self.dense_layer.append(layer)
            n_channels = layer.n_channels

            # transition layer
            layer = _TransLayer(self, n_channels, i==self.n_blocks-1)
            self.trans_layer.append(layer)
            n_channels = layer.n_channels

        self.output_layer = _OutputLayer(self, n_channels)

    def forward(self, x):
        y = self.input_layer(x)
        for i in range(self.n_blocks):
            y = self.dense_layer[i](y)
            y = self.trans_layer[i](y)
        y = self.output_layer(y)
        return y
