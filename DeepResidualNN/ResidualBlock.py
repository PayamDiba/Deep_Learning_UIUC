"""
@author: Payam Dibaeinia
"""

import torch
import torch.nn as nn
from collections import OrderedDict


class BasicBlock(nn.Module):

    def __init__(self,in_channels, out_channels, stride = 1, downSample = False):
        super(BasicBlock,self).__init__()
        self.downSample_ = downSample
        self.conv_bn1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = stride, padding = 1)),
            ('bn1', nn.BatchNorm2d(out_channels)),
        ]))

        self.conv_bn2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1)),
            ('bn2', nn.BatchNorm2d(out_channels)),
        ]))

        self.relu = nn.ReLU(inplace = True)

        if downSample:
            self.conv_bn_DS = nn.Sequential(OrderedDict([
                ('convDS', nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = stride, padding = 0)),
                ('bnDS', nn.BatchNorm2d(out_channels)),
            ]))


    def forward(self,x):
        residual = x

        ret = self.conv_bn1(x)
        ret = self.relu(ret)
        ret = self.conv_bn2(ret)

        if self.downSample_:
            residual = self.conv_bn_DS(residual)

        ret += residual
        ret = self.relu(ret)
        return ret
