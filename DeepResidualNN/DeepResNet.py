"""
@author: Payam Dibaeinia
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from ResidualBlock import BasicBlock


class ResNet(nn.Module):
    """
    Build a residual network model. By default assumes there are 4 and only 4 basic blocks.
    """

    def __init__(self, output_size: int, block_sizes: list = [2,4,4,2], block_strides: list = [1,2,2,2]):
        super(ResNet, self).__init__()

        self.conv_preBlock = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)),
            ('bn', nn.BatchNorm2d(32)),
            ('relu', nn.ReLU(inplace = True)),
            ('dropout', nn.Dropout2d(p = 0.25))
        ]))

        self.block1 = self._stackBlocks(32,32,block_sizes[0], block_strides[0])
        self.block2 = self._stackBlocks(32,64,block_sizes[1], block_strides[1])
        self.block3 = self._stackBlocks(64,128,block_sizes[2], block_strides[2])
        self.block4 = self._stackBlocks(128,256,block_sizes[3], block_strides[3])
        self.maxPool = nn.MaxPool2d(kernel_size=4, stride=1)
        self.fc = nn.Linear(256, output_size)


    def _stackBlocks(self, in_channels, out_channels, nBlock, stride):
        """
        Note1: the stride parameter is only used for the fist conv_bn sub-block

        Note2: We need to check if down sampling is required. It's either required on
        the very first sublock or not required at all because both of the stride is
        always 1 for subblock 2 to ... and number of channels is conserved between subblocks

        Note3: Down sampling is required if stride > 1 or if in_channels != out_channels
        """

        if in_channels != out_channels or stride > 1:
            downSample = True
        else:
            downSample = False

        blocks = []
        blocks.append(BasicBlock(in_channels = in_channels, out_channels = out_channels, stride = stride, downSample = downSample))

        for i in range(1,nBlock):
            blocks.append(BasicBlock(in_channels = out_channels, out_channels = out_channels, stride = 1, downSample = False))

        ret = nn.Sequential(*blocks)
        return ret

    def forward(self,x):
        x = self.conv_preBlock(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.maxPool(x)
        x_flatten = x.view(x.size(0),-1)
        ret = self.fc(x_flatten)

        return ret
