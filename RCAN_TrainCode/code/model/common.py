import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias_attr=True):
    return nn.Conv2D(
        in_channels, out_channels, kernel_size,
        padding=kernel_size//2, bias_attr=bias_attr)

class MeanShift(nn.Conv2D):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = paddle.to_tensor(rgb_std)
        weight = paddle.eye(3).reshape([3, 3, 1, 1]).detach().divide(std.reshape([3, 1, 1, 1]))
        self.weight = paddle.create_parameter(shape=weight.shape,
                        dtype=str(weight.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Assign(weight))
        self.weight.stop_gradient = True
        # self.weight.divide(std.reshape([3, 1, 1, 1]))
        bias = sign * rgb_range * paddle.to_tensor(rgb_mean).detach().divide(std)
        self.bias = paddle.create_parameter(shape=bias.shape,
                        dtype=str(bias.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Assign(bias))
        self.bias.stop_gradient = True
        # self.bias.detach().divide(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias_attr=False,
        bn=True, act=nn.ReLU()):

        m = [nn.Conv2D(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias_attr=bias_attr)
        ]
        if bn: m.append(nn.BatchNorm2D(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Layer):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias_attr=True, bn=False, act=nn.ReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias_attr=bias_attr))
            if bn: m.append(nn.BatchNorm2D(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)*(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias_attr=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias_attr))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2D(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias_attr))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2D(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
